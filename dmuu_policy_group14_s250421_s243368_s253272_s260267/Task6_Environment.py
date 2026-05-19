# -*- coding: utf-8 -*-
"""
Task6_Environment.py

Implements the policy-evaluation loop from the assignment diagram:

    for t in range(T):
        state      = next_state
        decisions  = policy(state)                          # call as a plain function
        if not check_feasibility(decisions):
            decisions = dummy_action(state)                 # mandatory fallback
        cost[t]    = price * total_power_from_grid(...)    # after overrule controllers
        next_state = apply_dynamics(state, decisions)

Public API
----------
    dummy_action(state)                              -> decisions dict
    check_feasibility(decisions)                     -> bool
    apply_dynamics(state, decisions)                 -> (next_state, step_cost)
    run_policy(policy_fn, price_csv, occ1_csv,
               occ2_csv, ...)                        -> (daily_costs, solve_times_ms)
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from Data.v2_SystemCharacteristics import get_fixed_data


# ==============================================================================
# SYSTEM PARAMETERS  (loaded once at import time)
# ==============================================================================
_DATA = get_fixed_data()
_P = {
    'P_max'    : _DATA['heating_max_power'],
    'P_vent'   : _DATA['ventilation_power'],
    'T_low'    : _DATA['temp_min_comfort_threshold'],
    'T_OK'     : _DATA['temp_OK_threshold'],
    'T_high'   : _DATA['temp_max_comfort_threshold'],
    'H_high'   : _DATA['humidity_threshold'],
    'zeta_exch': _DATA['heat_exchange_coeff'],
    'zeta_loss': _DATA['thermal_loss_coeff'],
    'zeta_conv': _DATA['heating_efficiency_coeff'],
    'zeta_cool': _DATA['heat_vent_coeff'],
    'zeta_occ' : _DATA['heat_occupancy_coeff'],
    'eta_occ'  : _DATA['humidity_occupancy_coeff'],
    'eta_vent' : _DATA['humidity_vent_coeff'],
    'T_out'    : _DATA['outdoor_temperature'],
}
_T_INIT = float(_DATA['T1'])   # initial room temperature (°C)
_H_INIT = float(_DATA['H'])    # initial humidity (%)



# ==============================================================================
# BUILDING BLOCK 1 — dummy_action(state)
# ==============================================================================

def dummy_action(state: dict) -> dict:
    """
    Reactive baseline: only takes mandatory overrule actions, no optimisation.

    Heats at P_max only when y_low_r = 1 (cold override active).
    Ventilates only when c > 0 (inertia) or H >= H_high (humidity override).
    Used as fallback when the main policy returns an infeasible decision.
    """
    p = _P
    return {
        'HeatPowerRoom1': p['P_max'] if state.get('low_override_r1') else 0.0,
        'HeatPowerRoom2': p['P_max'] if state.get('low_override_r2') else 0.0,
        'VentilationON' : 1 if (int(state.get('vent_counter', 0)) > 0
                                or float(state.get('H', 0.0)) >= p['H_high']) else 0,
    }


# ==============================================================================
# BUILDING BLOCK 2 — check_feasibility(decisions)
# ==============================================================================

def check_feasibility(decisions: dict) -> bool:
    """
    Return True if all decision values lie within physical bounds.

    Bounds: 0 <= p1, p2 <= P_max;  v in {0, 1}.
    """
    p  = _P
    try:
        p1 = float(decisions['HeatPowerRoom1'])
        p2 = float(decisions['HeatPowerRoom2'])
        v  = int(float(decisions['VentilationON']))
    except (KeyError, TypeError, ValueError):
        return False
    return (
        0.0 <= p1 <= p['P_max'] and
        0.0 <= p2 <= p['P_max'] and
        v in (0, 1)
    )


# ==============================================================================
# BUILDING BLOCK 3 — apply_dynamics(state, decisions)
# ==============================================================================

def apply_dynamics(state: dict, decisions: dict) -> tuple:
    """
    Apply one time step.

    Steps (in order):
      1. Enforce overrule controllers (humidity, inertia, temperature limits).
      2. Compute step cost = price × (p1 + p2 + P_vent × v).
      3. Advance thermal / humidity state.
      4. Update ventilation inertia counter and temperature-override flags.

    Parameters
    ----------
    state     : dict  Current state (policy-facing keys: T1, T2, H, c,
                      y_low_1, y_low_2, y_high_1, y_high_2, occ1, occ2,
                      price, price_previous, current_time).
    decisions : dict  Raw policy output (HeatPowerRoom1, HeatPowerRoom2,
                      VentilationON). Overrule controllers are applied here.

    Returns
    -------
    next_state : dict  (same key set as state; occ/price are carried forward
                        and will be overwritten by CSV data at the next step)
    step_cost  : float  Actual electricity cost for this step (euro).
    """
    p   = _P
    t   = int(state['current_time'])
    p1  = float(decisions['HeatPowerRoom1'])
    p2  = float(decisions['HeatPowerRoom2'])
    v   = int(float(decisions['VentilationON']) > 0.5)

    # ── Overrule controllers ──────────────────────────────────────────────────
    H = float(state.get('H', 0.0))
    c = int(state.get('vent_counter', 0))
    if c > 0:              v  = 1               # ventilation min-up-time
    if H >= p['H_high']:   v  = 1               # humidity override
    if state.get('low_override_r1'):  p1 = p['P_max']   # cold room 1
    if state.get('low_override_r2'):  p2 = p['P_max']   # cold room 2
    if state.get('y_high_1'): p1 = 0.0          # too hot room 1
    if state.get('y_high_2'): p2 = 0.0          # too hot room 2

    # ── cost[t] = price × total_power_from_grid ───────────────────────────────
    step_cost = float(state['price_t']) * (p1 + p2 + p['P_vent'] * v)

    # ── Thermal / humidity dynamics ───────────────────────────────────────────
    # T_out is always taken from the deterministic schedule (known in advance).
    T_out = float(p['T_out'][min(t, 9)])
    T1    = float(state['T1'])
    T2    = float(state['T2'])
    occ1  = float(state.get('Occ1', 0.0))
    occ2  = float(state.get('Occ2', 0.0))

    T1_n = (T1
            + p['zeta_exch'] * (T2    - T1)
            + p['zeta_loss'] * (T_out - T1)
            + p['zeta_conv'] * p1
            - p['zeta_cool'] * v
            + p['zeta_occ']  * occ1)
    T2_n = (T2
            + p['zeta_exch'] * (T1    - T2)
            + p['zeta_loss'] * (T_out - T2)
            + p['zeta_conv'] * p2
            - p['zeta_cool'] * v
            + p['zeta_occ']  * occ2)
    H_n  = max(0.0, H + p['eta_occ'] * (occ1 + occ2) - p['eta_vent'] * v)

    # ── Ventilation inertia counter (decrementing: 2 → 1 → 0) ────────────────
    if v == 1:
        c_n = 2 if c == 0 else max(0, c - 1)
    else:
        c_n = 0

    # ── Temperature override flags ────────────────────────────────────────────
    y_lo1 = (1 if T1_n < p['T_low']
             else (0 if T1_n >= p['T_OK'] else int(state.get('low_override_r1', 0))))
    y_lo2 = (1 if T2_n < p['T_low']
             else (0 if T2_n >= p['T_OK'] else int(state.get('low_override_r2', 0))))
    y_hi1 = 1 if T1_n > p['T_high'] else 0
    y_hi2 = 1 if T2_n > p['T_high'] else 0

    next_state = {
        'T1': T1_n, 'T2': T2_n, 'H': H_n,
        'vent_counter': c_n,
        'low_override_r1': y_lo1, 'low_override_r2': y_lo2,
        'y_high_1': y_hi1, 'y_high_2': y_hi2,
        # occ / price are placeholders; overwritten by CSV at the next step
        'Occ1'          : occ1,
        'Occ2'          : occ2,
        'price_t'       : float(state['price_t']),
        'price_previous': float(state['price_previous']),
        'current_time'  : t + 1,
    }
    return next_state, step_cost


# ==============================================================================
# MAIN EVALUATION LOOP — run_policy(policy_fn, price_csv, occ1_csv, occ2_csv, ...)
# ==============================================================================

def run_policy(policy_fn, price_csv, occ1_csv, occ2_csv,
               policy_name: str = "Policy",
               num_days: int = None, verbose: bool = False):
    """
    Evaluate a policy directly from the three source CSV files.

    Implements the assignment evaluation loop exactly:

        for t in range(T):
            state      = next_state
            decisions  = policy_fn(state)
            if not check_feasibility(decisions):
                decisions = dummy_action(state)
            cost[t]    = price × total_power_from_grid(state, decisions)
            next_state = apply_dynamics(state, decisions)

    Parameters
    ----------
    policy_fn   : callable
        ``policy_fn(state: dict) -> dict`` returning
        {HeatPowerRoom1, HeatPowerRoom2, VentilationON}.
    price_csv   : str or Path  Path to v2_PriceData.csv.
    occ1_csv    : str or Path  Path to OccupancyRoom1.csv.
    occ2_csv    : str or Path  Path to OccupancyRoom2.csv.
    policy_name : str          Label used in verbose output.
    num_days    : int or None  Number of days to evaluate (None = all).
    verbose     : bool         Print per-day cost and average step timing.

    Returns
    -------
    daily_costs : np.ndarray (num_days,)    daily electricity cost (euro)
    solve_times : np.ndarray (num_days*10,) wall-clock time per step (ms)
    """
    price_df = pd.read_csv(price_csv, header=0)
    occ1_df  = pd.read_csv(occ1_csv,  header=0)
    occ2_df  = pd.read_csv(occ2_csv,  header=0)

    price_cols = [str(i) for i in range(1, 11)]
    occ_cols   = [str(i) for i in range(10)]

    prices_matrix = price_df[price_cols].values        # (n_days, 10)
    prev_prices   = price_df.iloc[:, 0].values         # (n_days,) previous price before slot 0
    occ1_matrix   = occ1_df[occ_cols].values           # (n_days, 10)
    occ2_matrix   = occ2_df[occ_cols].values           # (n_days, 10)

    n_days = len(prices_matrix)
    if num_days is not None:
        n_days = min(int(num_days), n_days)

    daily_costs = np.zeros(n_days)
    solve_times = []

    for di in range(n_days):
        prices = prices_matrix[di]    # (10,)
        occ1s  = occ1_matrix[di]      # (10,)
        occ2s  = occ2_matrix[di]      # (10,)

        # ── Fixed initial physical state ──────────────────────────────────────
        state = {
            'T1': _T_INIT, 'T2': _T_INIT, 'H': _H_INIT,
            'vent_counter': 0, 'low_override_r1': 0, 'low_override_r2': 0,
            'y_high_1': 0, 'y_high_2': 0,
            'Occ1'          : float(occ1s[0]),
            'Occ2'          : float(occ2s[0]),
            'price_t'       : float(prices[0]),
            'price_previous': float(prev_prices[di]),
            'current_time'  : 0,
        }

        for t in range(10):
            # ── Inject realized values for this hour ──────────────────────────
            state['price_t']        = float(prices[t])
            state['price_previous'] = float(prices[t - 1]) if t > 0 else float(prev_prices[di])
            state['Occ1']           = float(occ1s[t])
            state['Occ2']           = float(occ2s[t])
            state['current_time']   = t
            # T_out is NOT injected — apply_dynamics uses the fixed schedule

            # ── decisions = policy(state) ─────────────────────────────────────
            t0 = time.perf_counter()
            try:
                decisions = policy_fn(state)
            except Exception:
                decisions = dummy_action(state)
            solve_times.append((time.perf_counter() - t0) * 1000)

            # ── Infeasible → fallback to dummy_action(state) ──────────────────
            if not check_feasibility(decisions):
                decisions = dummy_action(state)

            # ── cost[t], next_state = apply_dynamics(state, decisions) ─────────
            state, step_cost = apply_dynamics(state, decisions)
            daily_costs[di] += step_cost

        if verbose:
            avg_ms = np.mean(solve_times[-10:])
            print(f"  Day {di+1:>3}/{n_days}"
                  f"  |  cost = {daily_costs[di]:7.2f} euro"
                  f"  |  avg step = {avg_ms:.0f} ms")

    st = np.array(solve_times)
    if verbose:
        print(f"\nDone — {n_days} days × 10 steps"
              f"  |  mean {st.mean():.0f} ms  |  max {st.max():.0f} ms")
        print(f"{policy_name}  —  mean daily cost: "
              f"{daily_costs.mean():.3f} euro  (std {daily_costs.std():.3f})")

    return daily_costs, st
