"""
ADP_policy_14.py
================
Approximate Dynamic Programming (ADP) policy — Group 14, DTU 02435, Spring 2026.

Architecture
------------
Single-step MILP that minimises the immediate energy cost plus a linear Value
Function Approximation (VFA) terminal cost trained offline by ADP_Training_14.py.

    min  price * (p1 + p2 + P_vent * v)              [immediate energy cost]
       + E[ V_{t+1}(T1', T2', H', c', ov1', ov2',
                    price_next, occ1_next, occ2_next) ]  [VFA terminal cost]

The VFA approximates the expected cost-to-go from the next state.
Stochastic future prices and occupancies are pre-sampled as a constant
(K_POLICY Monte Carlo draws) so the MILP remains linear.

VFA features (normalised, from ADP_weights.csv):
    T1_feat          = (T1_next - 22) / 8
    T2_feat          = (T2_next - 22) / 8
    H_feat           = (H_next  - 40) / 40
    vc_feat          = vent_counter_next / 3
    ov1_feat, ov2_feat  low-temperature override flags (binary)
    price_prev_feat  = price_t / 10
    E[price_t_next/10 + (Occ1_next-20)/30 + (Occ2_next-10)/20]  [MC constant]

Training:   ADP_Training_14.py   (Approximate Policy Iteration, Ridge regression)
Weights:    Data/ADP_weights.csv

Public interface:
    select_action(state) -> {'HeatPowerRoom1', 'HeatPowerRoom2', 'VentilationON'}
"""

import csv
from pathlib import Path
from pyomo.environ import *
from Data.v2_SystemCharacteristics   import get_fixed_data
from Data.PriceProcessRestaurant     import price_model
from Data.OccupancyProcessRestaurant import next_occupancy_levels

# ── System parameters (loaded once at import time) ────────────────────────────
_SYS = get_fixed_data()
_P = {
    'P_max'    : _SYS['heating_max_power'],
    'P_vent'   : _SYS['ventilation_power'],
    'T_low'    : _SYS['temp_min_comfort_threshold'],
    'T_OK'     : _SYS['temp_OK_threshold'],
    'T_high'   : _SYS['temp_max_comfort_threshold'],
    'H_high'   : _SYS['humidity_threshold'],
    'zeta_exch': _SYS['heat_exchange_coeff'],
    'zeta_loss': _SYS['thermal_loss_coeff'],
    'zeta_conv': _SYS['heating_efficiency_coeff'],
    'zeta_cool': _SYS['heat_vent_coeff'],
    'zeta_occ' : _SYS['heat_occupancy_coeff'],
    'eta_occ'  : _SYS['humidity_occupancy_coeff'],
    'eta_vent' : _SYS['humidity_vent_coeff'],
    'T_out'    : _SYS['outdoor_temperature'],
}

# ── ADP weights (trained by ADP_Training_14.py) ───────────────────────────────
def _load_adp_weights():
    csv_path = Path(__file__).resolve().parent / "Data" / "ADP_weights.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"ADP weights not found: {csv_path}")
    weights = {}
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = int(row["t"])
            weights[t] = {k: float(v) for k, v in row.items() if k != "t"}
    return weights

_ADP_WEIGHTS = _load_adp_weights()

# Monte Carlo draws to pre-compute the stochastic VFA constant
_K_POLICY = 15


def select_action(state: dict) -> dict:
    """
    ADP policy: single-step MILP + VFA terminal cost.

    Parameters
    ----------
    state : dict
        Keys: T1, T2, H, Occ1, Occ2, price_t, price_previous,
              vent_counter, low_override_r1, low_override_r2, current_time.

    Returns
    -------
    dict with keys HeatPowerRoom1, HeatPowerRoom2, VentilationON.
    """
    p    = _P
    t    = int(state['current_time'])
    c    = int(state.get('vent_counter', 0))
    T1   = float(state['T1'])
    T2   = float(state['T2'])
    H    = float(state['H'])
    occ1 = float(state['Occ1'])
    occ2 = float(state['Occ2'])
    price      = float(state['price_t'])
    price_prev = float(state.get('price_previous', price))
    y_lo1 = int(state.get('low_override_r1', 0))
    y_lo2 = int(state.get('low_override_r2', 0))
    T_out = float(p['T_out'][min(t, 9)])

    M, eps = 500, 1e-5

    m = ConcreteModel()

    # ── Decision variables ─────────────────────────────────────────────────────
    m.p1 = Var(bounds=(0.0, p['P_max']))
    m.p2 = Var(bounds=(0.0, p['P_max']))
    m.v  = Var(domain=Binary)

    # ── Next physical state variables ──────────────────────────────────────────
    m.T1_next          = Var()
    m.T2_next          = Var()
    m.H_next           = Var()
    m.vent_counter_next = Var(domain=NonNegativeReals)

    # ── Overrule controllers (hard constraints matching Task6_Environment) ─────
    if T1 > p['T_high']:              m.p1.fix(0.0)
    elif y_lo1 and T1 < p['T_OK']:    m.p1.fix(p['P_max'])
    elif T1 < p['T_low']:             m.p1.fix(p['P_max'])

    if T2 > p['T_high']:              m.p2.fix(0.0)
    elif y_lo2 and T2 < p['T_OK']:    m.p2.fix(p['P_max'])
    elif T2 < p['T_low']:             m.p2.fix(p['P_max'])

    if H > p['H_high'] or c in (1, 2): m.v.fix(1)

    # ── Thermal and humidity dynamics ──────────────────────────────────────────
    m.c_T1 = Constraint(expr=
        m.T1_next == T1
        + p['zeta_exch'] * (T2  - T1)
        + p['zeta_loss'] * (T_out - T1)
        + p['zeta_conv'] * m.p1
        - p['zeta_cool'] * m.v
        + p['zeta_occ']  * occ1)

    m.c_T2 = Constraint(expr=
        m.T2_next == T2
        + p['zeta_exch'] * (T1  - T2)
        + p['zeta_loss'] * (T_out - T2)
        + p['zeta_conv'] * m.p2
        - p['zeta_cool'] * m.v
        + p['zeta_occ']  * occ2)

    m.c_H = Constraint(expr=
        m.H_next == H
        + p['eta_occ']  * (occ1 + occ2)
        - p['eta_vent'] * m.v)

    # ── Ventilation counter after this step ────────────────────────────────────
    if c == 0:
        m.c_vc = Constraint(expr=m.vent_counter_next == 2 * m.v)
    elif c == 1:
        m.vent_counter_next.fix(0)
    else:   # c == 2
        m.vent_counter_next.fix(1)

    # ── Low-temperature override flags for the next state (Big-M) ─────────────
    # Room 1
    m.z_low_r1 = Var(domain=Binary)   # 1 if T1_next < T_low
    m.z_ok_r1  = Var(domain=Binary)   # 1 if T1_next >= T_OK
    m.ov1_next = Var(domain=Binary)   # resulting override flag for room 1

    m.c_zlow_r1_a = Constraint(expr=m.T1_next <= p['T_low'] + M*(1 - m.z_low_r1))
    m.c_zlow_r1_b = Constraint(expr=m.T1_next >= p['T_low'] + eps - M*m.z_low_r1)
    m.c_zok_r1_a  = Constraint(expr=m.T1_next >= p['T_OK']  - M*(1 - m.z_ok_r1))
    m.c_zok_r1_b  = Constraint(expr=m.T1_next <= p['T_OK']  + M*m.z_ok_r1)

    # ov1_next = 1 iff (T1_next < T_low) OR (y_lo1 AND T1_next < T_OK)
    m.c_ov1_on_if_cold   = Constraint(expr=m.ov1_next >= m.z_low_r1)
    m.c_ov1_bounded_above = Constraint(expr=m.ov1_next <= y_lo1 + m.z_low_r1)
    m.c_ov1_persist      = Constraint(expr=m.ov1_next >= y_lo1 - m.z_ok_r1)
    m.c_ov1_off_if_ok    = Constraint(expr=m.ov1_next <= 1 - m.z_ok_r1)

    # Room 2
    m.z_low_r2 = Var(domain=Binary)
    m.z_ok_r2  = Var(domain=Binary)
    m.ov2_next = Var(domain=Binary)

    m.c_zlow_r2_a = Constraint(expr=m.T2_next <= p['T_low'] + M*(1 - m.z_low_r2))
    m.c_zlow_r2_b = Constraint(expr=m.T2_next >= p['T_low'] + eps - M*m.z_low_r2)
    m.c_zok_r2_a  = Constraint(expr=m.T2_next >= p['T_OK']  - M*(1 - m.z_ok_r2))
    m.c_zok_r2_b  = Constraint(expr=m.T2_next <= p['T_OK']  + M*m.z_ok_r2)

    m.c_ov2_on_if_cold    = Constraint(expr=m.ov2_next >= m.z_low_r2)
    m.c_ov2_bounded_above = Constraint(expr=m.ov2_next <= y_lo2 + m.z_low_r2)
    m.c_ov2_persist       = Constraint(expr=m.ov2_next >= y_lo2 - m.z_ok_r2)
    m.c_ov2_off_if_ok     = Constraint(expr=m.ov2_next <= 1 - m.z_ok_r2)

    # ── Immediate cost ─────────────────────────────────────────────────────────
    immediate_cost = price * (m.p1 + m.p2 + p['P_vent'] * m.v)

    # ── VFA terminal cost (if not the last step) ───────────────────────────────
    if t < 9:
        w = _ADP_WEIGHTS[t + 1]

        # Pre-sample stochastic next price and occupancy (constant in MILP)
        sc_price = [price_model(price, price_prev)           for _ in range(_K_POLICY)]
        sc_occ   = [next_occupancy_levels(occ1, occ2)        for _ in range(_K_POLICY)]
        stoch_const = (1.0 / _K_POLICY) * sum(
              w['price_t'] * (sc_price[k]         / 10.0)
            + w['Occ1']    * ((sc_occ[k][0] - 20.0) / 30.0)
            + w['Occ2']    * ((sc_occ[k][1] - 10.0) / 20.0)
            for k in range(_K_POLICY)
        )

        expected_future_cost = (
            w['intercept']
            + w['T1']              * ((m.T1_next          - 22.0) / 8.0)
            + w['T2']              * ((m.T2_next          - 22.0) / 8.0)
            + w['H']               * ((m.H_next           - 40.0) / 40.0)
            + w['vent_counter']    * (m.vent_counter_next  / 3.0)
            + w['low_override_r1'] *  m.ov1_next
            + w['low_override_r2'] *  m.ov2_next
            + w['price_previous']  * (price / 10.0)
            + stoch_const
        )
    else:
        expected_future_cost = 0.0

    m.obj = Objective(expr=immediate_cost + expected_future_cost, sense=minimize)

    solver = SolverFactory('gurobi')
    solver.options['OutputFlag'] = 0
    solver.solve(m)

    return {
        'HeatPowerRoom1': float(value(m.p1)),
        'HeatPowerRoom2': float(value(m.p2)),
        'VentilationON' : int(round(float(value(m.v)))),
    }
