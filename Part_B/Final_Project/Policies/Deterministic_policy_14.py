"""
Deterministic_policy_14.py
==========================
Deterministic lookahead (MPC / Expected-Value) policy — Group 14, DTU 02435, Spring 2026.

At each step t the policy solves a full-horizon MILP using:
  - ACTUAL observed price and occupancy for the CURRENT step t (from the MDP state).
  - HISTORICAL MEAN price and occupancy (computed from the CSV pool) for all
    FUTURE steps t+1 .. 9.

This is the Expected-Value (EV) approximation of the stochastic problem: it
replaces uncertain future disturbances with their sample means and solves the
resulting deterministic problem exactly.  It is computationally cheap (one MILP
per step, no scenario sampling) and serves as a strong deterministic baseline.

The MILP is structurally identical to the hindsight formulation: full Big-M
overrule modelling, ventilation min-up-time, same dynamics.  Only the first
action of the plan is executed (receding horizon / MPC).

Public interface (same as all other Group 14 policies):
    select_action(state: dict) -> {'HeatPowerRoom1', 'HeatPowerRoom2', 'VentilationON'}
"""

import os
import sys
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
for _d in [_dir, os.path.join(_dir, 'Data')]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

# Reuse the full hindsight MILP formulation and system parameters
from Policies.hindsight_optimization import _solve_day, _P
from Data.PriceProcessRestaurant import price_model
from Data.OccupancyProcessRestaurant import next_occupancy_levels

# ── Compute slot-wise means from 100 simulated scenarios ─────────────────────
_N_SCEN = 100
_T      = 10

_rng = np.random.default_rng(seed=42)   # fixed seed → reproducible means

_price_scens = np.zeros((_N_SCEN, _T))
_occ1_scens  = np.zeros((_N_SCEN, _T))
_occ2_scens  = np.zeros((_N_SCEN, _T))

for _s in range(_N_SCEN):
    # ── Price trajectory ──────────────────────────────────────────────────────
    _p = [float(_rng.uniform(2, 8))]
    for _t in range(1, _T):
        _prev_prev = _p[-2] if _t > 1 else 6.0
        _p.append(price_model(_p[-1], _prev_prev))
    _price_scens[_s] = _p

    # ── Occupancy trajectories ────────────────────────────────────────────────
    _r1 = float(_rng.uniform(25, 35))
    _r2 = float(_rng.uniform(15, 25))
    _occ1_scens[_s, 0] = _r1
    _occ2_scens[_s, 0] = _r2
    for _t in range(1, _T):
        _r1, _r2 = next_occupancy_levels(_r1, _r2)
        _occ1_scens[_s, _t] = _r1
        _occ2_scens[_s, _t] = _r2

_MEAN_PRICES = _price_scens.mean(axis=0)   # (10,)
_MEAN_OCC1   = _occ1_scens.mean(axis=0)    # (10,)
_MEAN_OCC2   = _occ2_scens.mean(axis=0)    # (10,)


def _dummy_action(state):
    c = int(state.get('vent_counter', 0))
    H = float(state.get('H', 0.0))
    return {
        'HeatPowerRoom1': _P['P_max'] if state.get('low_override_r1') else 0.0,
        'HeatPowerRoom2': _P['P_max'] if state.get('low_override_r2') else 0.0,
        'VentilationON' : 1 if (c > 0 or H >= _P['H_high']) else 0,
    }


def select_action(state: dict) -> dict:
    """
    Deterministic lookahead (EV-MPC) policy.

    Builds a 10-element price/occ array where index 0 is the observed current
    value and indices 1..(9-t) are historical slot means.  Solves the full-
    horizon hindsight MILP on this deterministic forecast and returns the first
    (here-and-now) action.
    """
    t = int(state.get('current_time', 0))
    c = int(state.get('vent_counter', 0))
    n_rem = 10 - t   # number of remaining steps including current

    # ── Deterministic forecast arrays (padded to 10 for _solve_day) ──────────
    prices_det = np.zeros(10)
    occ1_det   = np.zeros(10)
    occ2_det   = np.zeros(10)

    # Slot 0 (MILP index) = current actual step
    prices_det[0] = float(state['price_t'])
    occ1_det[0]   = float(state['Occ1'])
    occ2_det[0]   = float(state['Occ2'])

    # Slots 1 .. n_rem-1 = expected values from historical means
    for i in range(1, n_rem):
        prices_det[i] = _MEAN_PRICES[t + i]
        occ1_det[i]   = _MEAN_OCC1[t + i]
        occ2_det[i]   = _MEAN_OCC2[t + i]
    # Slots n_rem .. 9 remain zero (zero-price dummy steps, don't affect t=0 action)

    # T_out: fully deterministic schedule — map MILP slot k → actual slot t+k
    T_outs_det = np.array([float(_P['T_out'][min(t + k, 9)]) for k in range(10)])

    # ── Solve deterministic MILP for the remaining horizon ────────────────────
    try:
        result = _solve_day(
            prices   = prices_det,
            occ1s    = occ1_det,
            occ2s    = occ2_det,
            T_outs   = T_outs_det,
            T1_init  = float(state.get('T1', _P['T1_init'])),
            T2_init  = float(state.get('T2', _P['T2_init'])),
            H_init   = float(state.get('H',  _P['H_init'])),
            c0       = c,
            y_low1_0 = int(state.get('low_override_r1', 0)),
            y_low2_0 = int(state.get('low_override_r2', 0)),
        )
    except Exception:
        return _dummy_action(state)

    if not result['feasible'] or result['actions'] is None:
        return _dummy_action(state)

    # Execute only the first action of the plan (receding horizon)
    p1, p2, v = result['actions'][0]
    return {
        'HeatPowerRoom1': float(p1),
        'HeatPowerRoom2': float(p2),
        'VentilationON' : int(round(float(v))),
    }
