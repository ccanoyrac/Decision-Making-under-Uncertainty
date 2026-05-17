"""
hindsight_optimization.py
=========================
Solves the 10-step perfect-information (hindsight-optimal) MILP for every day
in the input CSVs, giving the provable lower bound on daily electricity cost.

Mathematical formulation — Task 1 of the assignment (Solution_to_Assignment_A_2026.pdf)
----------------------------------------------------------------------------------------
Objective (Eq. 1):
    min  sum_t  lambda_t * (P^vent * v_t + p1_t + p2_t)

Temperature dynamics (Eq. 2):
    T1_{t+1} = T1_t + zeta_exch*(T2_t - T1_t) + zeta_loss*(T_out_t - T1_t)
             + zeta_conv*p1_t - zeta_cool*v_t + zeta_occ*occ1_t     (same for T2)

Humidity dynamics (Eq. 3):
    H_{t+1} = H_t + eta_occ*(occ1_t + occ2_t) - eta_vent*v_t

Overrule controllers (explicitly modelled with binary indicator variables):
  Low-temperature overrule:
    y_loR[t] = 1  iff  T_Rx[t-1] <= T_low  OR  (y_loR[t-1]=1 AND T_Rx[t-1] < T_OK)
    y_loR[t] = 1  =>   pR[t] = P_max
  High-temperature overrule:
    T_Rx[t-1] > T_high  =>  pR[t] = 0
  Humidity overrule:
    Hx[t-1] > H_high    =>  v[t] = 1
  Ventilation minimum up-time (Eqs. 17-20):
    sum_{s=t}^{min(t+L-1, 9)} v_s  >=  min(L, 10-t) * (v_t - v_{t-1})
    where L = vent_min_up_time = 3

Indicator variables use Big-M linearisation with M_T=60 (temperature) and
M_H=200 (humidity).  State variables are free (no hard comfort bounds), so the
MILP can plan strategies where it is cheaper to trigger an overrule controller
than to heat or ventilate proactively.

Usage
-----
    from hindsight_optimization import run_hindsight

    daily_costs, details = run_hindsight(
        price_csv = "v2_PriceData.csv",
        occ1_csv  = "OccupancyRoom1.csv",
        occ2_csv  = "OccupancyRoom2.csv",
        verbose   = True,      # print per-day cost
    )
    # daily_costs : np.ndarray (n_days,)
    # details     : list of dicts with per-day cost, actions, temperatures
"""

import os, sys
import numpy as np
import pandas as pd
import pyomo.environ as pyo

# ── Load system parameters ────────────────────────────────────────────────────
_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

import v2_SystemCharacteristics as _sc

_RAW = _sc.get_fixed_data()
_P = {
    "T1_init" : _RAW['T1'],
    "T2_init" : _RAW['T2'],
    "H_init"  : _RAW['H'],
    "c0"      : _RAW['vent_counter'],
    "y_low1_0": _RAW['low_override_r1'],
    "y_low2_0": _RAW['low_override_r2'],
    'P_max'    : _RAW['heating_max_power'],
    'P_vent'   : _RAW['ventilation_power'],
    'T_low'    : _RAW['temp_min_comfort_threshold'],
    'T_OK'     : _RAW['temp_OK_threshold'],
    'T_high'   : _RAW['temp_max_comfort_threshold'],
    'H_high'   : _RAW['humidity_threshold'],
    'zeta_exch': _RAW['heat_exchange_coeff'],
    'zeta_loss': _RAW['thermal_loss_coeff'],
    'zeta_conv': _RAW['heating_efficiency_coeff'],
    'zeta_cool': _RAW['heat_vent_coeff'],
    'zeta_occ' : _RAW['heat_occupancy_coeff'],
    'eta_occ'  : _RAW['humidity_occupancy_coeff'],
    'eta_vent' : _RAW['humidity_vent_coeff'],
    'T_out'    : _RAW['outdoor_temperature'],
    'L'        : int(_RAW['vent_min_up_time']),   # = 3
}


# ── Single-day MILP ───────────────────────────────────────────────────────────
def _solve_day(prices, occ1s, occ2s, T_outs=_P['T_out'],
               T1_init=_P['T1_init'], T2_init=_P['T2_init'], H_init=_P['H_init'],
               c0=_P['c0'], y_low1_0=_P['y_low1_0'], y_low2_0=_P['y_low2_0']):
    """
    Solve the 10-step hindsight-optimal MILP for one day.

    Overrule controllers are modelled explicitly with binary indicator variables
    and Big-M linearisation — the MILP can therefore plan strategies that
    deliberately trigger an overrule (e.g. skip expensive heating now and let
    the low-temperature overrule force P_max at a cheaper future hour).

    Parameters
    ----------
    prices              : array-like (10,) — electricity price at each hour
    occ1s, occ2s        : array-like (10,) — room occupancies at each hour
    T_outs              : array-like (10,) — outdoor temperature at each hour
    T1_init, T2_init    : float — initial room temperatures (°C)
    H_init              : float — initial humidity (%)
    c0                  : int   — initial ventilation inertia counter
    y_low1_0, y_low2_0  : int   — initial low-temperature overrule flags (0/1)

    Returns
    -------
    dict with keys: 'cost', 'actions', 'temperatures', 'feasible'
    """
    p = _P
    T = list(range(10))
    L = p['L']
    v_prev_0 = 1 if c0 > 0 else 0

    # Big-M constants for linearisation
    M_T = 60.0    # °C — covers the full reachable temperature range
    M_H = 200.0   # %  — humidity stays well below this in practice

    # Overrule state of the system BEFORE step 0 (determined by initial state)
    y_hi1_0 = 1 if float(T1_init) > p['T_high'] else 0
    y_hi2_0 = 1 if float(T2_init) > p['T_high'] else 0
    # h_ov_0 no longer needed: z_hum[0] handles the t=0 humidity overrule automatically

    M = pyo.ConcreteModel()

    # ── Decision variables ────────────────────────────────────────────────────
    M.p1 = pyo.Var(T, bounds=(0.0, p['P_max']))
    M.p2 = pyo.Var(T, bounds=(0.0, p['P_max']))
    M.v  = pyo.Var(T, domain=pyo.Binary)

    # ── State variables (free — no hard comfort bounds) ───────────────────────
    M.T1x = pyo.Var(T, bounds=(-50.0, 100.0))
    M.T2x = pyo.Var(T, bounds=(-50.0, 100.0))
    M.Hx  = pyo.Var(T, bounds=(-200.0, 500.0))

    # ── Overrule indicator binary variables ───────────────────────────────────
    # y_loR[t] = 1: low-temp overrule ACTIVE at step t → pR[t] forced to P_max
    M.y_lo1 = pyo.Var(T, domain=pyo.Binary)
    M.y_lo2 = pyo.Var(T, domain=pyo.Binary)
    # z_bloR[t] = 1: T_Rx[t] ≤ T_low  (triggers y_loR at t+1)
    M.z_blo1 = pyo.Var(T, domain=pyo.Binary)
    M.z_blo2 = pyo.Var(T, domain=pyo.Binary)
    # z_bokR[t] = 1: T_Rx[t] ≤ T_OK   (persistence check: overrule continues until T_OK)
    M.z_bok1 = pyo.Var(T, domain=pyo.Binary)
    M.z_bok2 = pyo.Var(T, domain=pyo.Binary)
    # wR[t] = y_loR[t] AND z_bokR[t]  (linearised product for persistence logic)
    M.w1 = pyo.Var(T, domain=pyo.Binary)
    M.w2 = pyo.Var(T, domain=pyo.Binary)
    # z_ahiR[t] = 1: T_Rx[t] > T_high (triggers pR[t+1] = 0)
    M.z_ahi1 = pyo.Var(T, domain=pyo.Binary)
    M.z_ahi2 = pyo.Var(T, domain=pyo.Binary)
    # z_hum[t] = 1: Hx[t] > H_high    (triggers v[t] = 1, same step — matches environment)
    M.z_hum  = pyo.Var(T, domain=pyo.Binary)

    M.cons = pyo.ConstraintList()

    # Fix initial low-temp overrule flags (given as parameters)
    M.cons.add(M.y_lo1[0] == int(y_low1_0))
    M.cons.add(M.y_lo2[0] == int(y_low2_0))

    for t in T:
        T1p = float(T1_init) if t == 0 else M.T1x[t-1]
        T2p = float(T2_init) if t == 0 else M.T2x[t-1]
        Hp  = float(H_init)  if t == 0 else M.Hx[t-1]
        
        # Previous timestep actions and disturbances (for causal dynamics)
        # Actions and disturbances at t-1 affect state at t
        # At t=0, use current hour's disturbances (start of day assumption)
        if t == 0:
            Top = float(T_outs[0])
            o1p = float(occ1s[0])
            o2p = float(occ2s[0])
            p1_prev = 0.0
            p2_prev = 0.0
        else:
            Top = float(T_outs[t-1])
            o1p = float(occ1s[t-1])
            o2p = float(occ2s[t-1])
            p1_prev = M.p1[t-1]
            p2_prev = M.p2[t-1]
        
        v_prev  = v_prev_0 if t == 0 else M.v[t-1]

        # ── (1) Low-temperature overrule: y_lo=1 → p = P_max ──────────────
        M.cons.add(M.p1[t] >= p['P_max'] * M.y_lo1[t])
        M.cons.add(M.p2[t] >= p['P_max'] * M.y_lo2[t])

        # ── (2) High-temperature overrule: previous T > T_high → p = 0 ────
        if t == 0:
            if y_hi1_0: M.cons.add(M.p1[0] == 0.0)
            if y_hi2_0: M.cons.add(M.p2[0] == 0.0)
        else:
            # p1[t] ≤ P_max*(1 - z_ahi1[t-1]): if z=1 then p1=0
            M.cons.add(M.p1[t] <= p['P_max'] * (1 - M.z_ahi1[t-1]))
            M.cons.add(M.p2[t] <= p['P_max'] * (1 - M.z_ahi2[t-1]))

        # ── (3) Humidity overrule: current H >= H_high → v = 1 (same step) ──
        # Matches Task6_Environment: "if H >= H_high: v = 1" at the current step t.
        # z_hum[t] is defined below in the Big-M block; Pyomo adds constraints
        # symbolically so ordering within the loop body does not matter.
        # At t=0: Hx[0]=H_init is pinned, so z_hum[0]=1 automatically when H_init>H_high,
        # replacing the former h_ov_0 special case.
        M.cons.add(M.v[t] >= M.z_hum[t])

        # ── Dynamics (using previous timestep actions per assignment Eq. 2-3) ──
        if t != 0:
            M.cons.add(
                M.T1x[t] == T1p
                + p['zeta_exch'] * (T2p - T1p)
                + p['zeta_loss'] * (Top  - T1p)
                + p['zeta_conv'] * p1_prev
                - p['zeta_cool'] * v_prev
                + p['zeta_occ']  * o1p 
            )
            M.cons.add(
                M.T2x[t] == T2p
                + p['zeta_exch'] * (T1p - T2p)
                + p['zeta_loss'] * (Top  - T2p)
                + p['zeta_conv'] * p2_prev
                - p['zeta_cool'] * v_prev
                + p['zeta_occ']  * o2p
            )
            M.cons.add(
                M.Hx[t] == Hp
                + p['eta_occ']  * (o1p + o2p)
                - p['eta_vent'] * v_prev
            )
        else:
            # Initial state constraints at t=0
            M.cons.add(M.T1x[0] == T1p)
            M.cons.add(M.T2x[0] == T2p)
            M.cons.add(M.Hx[0]  == Hp)

        # ── Big-M indicator definitions ────────────────────────────────────
        # z_blo1[t] = 1  iff  T1x[t] ≤ T_low
        M.cons.add(M.T1x[t] >= p['T_low'] - M_T *  M.z_blo1[t])   # z=0 → T1x > T_low
        M.cons.add(M.T1x[t] <= p['T_low'] + M_T * (1-M.z_blo1[t]))# z=1 → T1x ≤ T_low
        M.cons.add(M.T2x[t] >= p['T_low'] - M_T *  M.z_blo2[t])
        M.cons.add(M.T2x[t] <= p['T_low'] + M_T * (1-M.z_blo2[t]))

        # z_bok1[t] = 1  iff  T1x[t] ≤ T_OK  (overrule persists below T_OK)
        M.cons.add(M.T1x[t] >= p['T_OK'] - M_T *  (1-M.z_bok1[t]))
        M.cons.add(M.T1x[t] <= p['T_OK'] + M_T * M.z_bok1[t])
        M.cons.add(M.T2x[t] >= p['T_OK'] - M_T *  (1-M.z_bok2[t]))
        M.cons.add(M.T2x[t] <= p['T_OK'] + M_T * M.z_bok2[t])

        # z_ahi1[t] = 1  iff  T1x[t] > T_high
        M.cons.add(M.T1x[t] <= p['T_high'] + M_T *  M.z_ahi1[t])   # z=0 → T1x ≤ T_high
        M.cons.add(M.T1x[t] >= p['T_high'] - M_T * (1-M.z_ahi1[t]))# z=1 → T1x > T_high
        M.cons.add(M.T2x[t] <= p['T_high'] + M_T *  M.z_ahi2[t])
        M.cons.add(M.T2x[t] >= p['T_high'] - M_T * (1-M.z_ahi2[t]))

        # z_hum[t] = 1  iff  Hx[t] > H_high
        M.cons.add(M.Hx[t]  <= p['H_high'] + M_H *  M.z_hum[t])    # z=0 → Hx ≤ H_high

        # ── Deactivating Overrule (only) if Temperature is above the "OK" level ──
        # Eq. (16): ur,t ≤ 1 - yOKr,t  →  if T ≥ T_OK then y_lo = 0
        # In our model: z_bok[t] = 0 when T > T_OK, so: y_lo[t] ≤ z_bok[t]
        M.cons.add(M.y_lo1[t] <= M.z_bok1[t])
        M.cons.add(M.y_lo2[t] <= M.z_bok2[t])

        # ── Low-temp overrule state propagation ────────────────────────────
        # y_lo1[t+1] = 1  iff  z_blo1[t]=1  OR  (y_lo1[t]=1 AND z_bok1[t]=1)
        if t < 9:
            # Linearise the AND: w1[t] = y_lo1[t] * z_bok1[t]
            M.cons.add(M.w1[t] <= M.y_lo1[t])
            M.cons.add(M.w1[t] <= M.z_bok1[t])
            M.cons.add(M.w1[t] >= M.y_lo1[t] + M.z_bok1[t] - 1)
            M.cons.add(M.w2[t] <= M.y_lo2[t])
            M.cons.add(M.w2[t] <= M.z_bok2[t])
            M.cons.add(M.w2[t] >= M.y_lo2[t] + M.z_bok2[t] - 1)
            # OR logic for y_lo propagation
            M.cons.add(M.y_lo1[t+1] >= M.z_blo1[t])             # fires if T ≤ T_low
            M.cons.add(M.y_lo1[t+1] >= M.w1[t])                 # persists if T < T_OK
            M.cons.add(M.y_lo1[t+1] <= M.z_blo1[t] + M.w1[t])  # off otherwise
            M.cons.add(M.y_lo2[t+1] >= M.z_blo2[t])
            M.cons.add(M.y_lo2[t+1] >= M.w2[t])
            M.cons.add(M.y_lo2[t+1] <= M.z_blo2[t] + M.w2[t])

        # ── Ventilation min-up-time ────────────────────────────────────────
        remaining = min(t + L, len(T)) - t
        v_prev_t  = v_prev_0 if t == 0 else M.v[t - 1]
        vsum      = sum(M.v[s] for s in range(t, t + remaining))
        M.cons.add(vsum >= remaining * (M.v[t] - v_prev_t))

    # Force ventilation ON for the first c0 hours (existing inertia)
    for tau in range(min(c0, len(T))):
        M.cons.add(M.v[tau] == 1)

    # ── Objective: minimise total electricity cost ────────────────────────────
    M.obj = pyo.Objective(
        expr=sum(
            float(prices[t]) * (M.p1[t] + M.p2[t] + p['P_vent'] * M.v[t])
            for t in T
        ),
        sense=pyo.minimize,
    )

    slv = pyo.SolverFactory('gurobi')
    slv.options['OutputFlag'] = 0
    slv.options['TimeLimit']  = 60
    res = slv.solve(M)

    ok = (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible)
    if res.solver.termination_condition not in ok:
        return {
            'cost': None, 'feasible': False,
            'actions': None, 'temperatures': None,
        }

    cost = sum(
        float(prices[t]) * (
            (pyo.value(M.p1[t]) or 0.0)
            + (pyo.value(M.p2[t]) or 0.0)
            + p['P_vent'] * (pyo.value(M.v[t]) or 0.0)
        )
        for t in T
    )
    actions = [
        (
            float(pyo.value(M.p1[t]) or 0.0),
            float(pyo.value(M.p2[t]) or 0.0),
            int(round(float(pyo.value(M.v[t]) or 0.0))),
        )
        for t in T
    ]
    temperatures = [
        (
            float(pyo.value(M.T1x[t]) or 0.0),
            float(pyo.value(M.T2x[t]) or 0.0),
            max(0.0, float(pyo.value(M.Hx[t]) or 0.0)),
        )
        for t in T
    ]
    return {
        'cost': cost, 'feasible': True,
        'actions': actions, 'temperatures': temperatures,
    }


# ── Main entry point ──────────────────────────────────────────────────────────
def run_hindsight(price_csv, occ1_csv, occ2_csv, verbose=False):
    """
    Run the hindsight-optimal MILP for every day in the input CSVs.

    Parameters
    ----------
    price_csv : str or Path
        Path to v2_PriceData.csv.  First column is the previous price at the
        first timeslot; columns named '1'–'10' are the electricity prices for
        the 10 hourly time slots of each day.
    occ1_csv : str or Path
        Path to OccupancyRoom1.csv.  Columns named '0'–'9' are room-1
        occupancies for each of the 10 time slots.
    occ2_csv : str or Path
        Path to OccupancyRoom2.csv.  Same layout as occ1_csv but for room 2.
    verbose : bool
        If True, print per-day optimal cost as it is computed.

    Returns
    -------
    daily_costs : np.ndarray of shape (n_days,)
        Optimal electricity cost for each day.  Infeasible days are set to NaN.
    details : list of dict
        Per-day breakdown with keys: day, cost, feasible, actions, temperatures.
    """
    price_df = pd.read_csv(price_csv, header=0)
    occ1_df  = pd.read_csv(occ1_csv,  header=0)
    occ2_df  = pd.read_csv(occ2_csv,  header=0)

    price_cols = [str(i) for i in range(1, 11)]
    occ_cols   = [str(i) for i in range(10)]

    prices_matrix = price_df[price_cols].values   # (n_days, 10)
    occ1_matrix   = occ1_df[occ_cols].values      # (n_days, 10)
    occ2_matrix   = occ2_df[occ_cols].values      # (n_days, 10)

    # Outdoor temperature: fixed sinusoidal profile from system characteristics,
    # identical for every day.
    T_outs = _P['T_out']

    n_days       = len(prices_matrix)
    daily_costs  = np.full(n_days, np.nan)
    details      = []
    n_infeasible = 0

    for di in range(n_days):
        prices = prices_matrix[di]
        occ1s  = occ1_matrix[di]
        occ2s  = occ2_matrix[di]

        # Initial physical state is fixed (deterministic from v2_SystemCharacteristics)
        result = _solve_day(
                prices, occ1s, occ2s, T_outs=_P['T_out'],
                T1_init=_P['T1_init'], T2_init=_P['T2_init'], H_init=_P['H_init'],
                c0=_P['c0'], y_low1_0=_P['y_low1_0'], y_low2_0=_P['y_low2_0'])

        result['day'] = di + 1

        if result['feasible']:
            daily_costs[di] = result['cost']
            if verbose:
                print(
                    f"  Day {di+1:>3}/{n_days}"
                    f"  |  optimal cost = {result['cost']:7.2f} euro"
                    f"  |  prices  [{prices.min():.2f}, {prices.max():.2f}]"
                )
        else:
            n_infeasible += 1
            if verbose:
                print(f"  Day {di+1:>3}/{n_days}  |  INFEASIBLE (solver failed)")

        details.append(result)

    valid = daily_costs[~np.isnan(daily_costs)]
    if verbose:
        print(f"\nDone — {n_days} days solved"
              f"  ({n_infeasible} infeasible, set to NaN)")
        if len(valid) > 0:
            print(f"Hindsight-optimal mean  : {valid.mean():.3f} euro")
            print(f"Hindsight-optimal std   : {valid.std():.3f} euro")
            print(f"Hindsight-optimal min   : {valid.min():.3f} euro")
            print(f"Hindsight-optimal max   : {valid.max():.3f} euro")

    return daily_costs, details


# ── CLI usage ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Hindsight-optimal MILP for HVAC cost minimisation.'
    )
    parser.add_argument('price_csv', help='Path to v2_PriceData.csv.')
    parser.add_argument('occ1_csv',  help='Path to OccupancyRoom1.csv.')
    parser.add_argument('occ2_csv',  help='Path to OccupancyRoom2.csv.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-day optimal cost.')
    args = parser.parse_args()

    costs, _ = run_hindsight(args.price_csv, args.occ1_csv, args.occ2_csv,
                             verbose=args.verbose)
    print(f"\nMean daily cost (hindsight-optimal): {np.nanmean(costs):.3f} euro")
