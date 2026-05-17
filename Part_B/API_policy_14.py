"""
API_policy_14.py
================
Deployment policy for the time-indexed Ridge VFA trained by ADP_Training_NEW.py.

Weights are loaded from api_vfa_weights.json (produced by training).
At each step t the policy solves a single-step MILP with a K_SCENARIOS
Monte-Carlo look-ahead using vfa_weights[t+1].

Public interface (same as all other Group 14 policies):
    select_action(state: dict) -> {'HeatPowerRoom1', 'HeatPowerRoom2', 'VentilationON'}
"""

import json
import os
import pyomo.environ as pyo
import numpy as np

from v2_SystemCharacteristics   import get_fixed_data
from PriceProcessRestaurant     import price_model
from OccupancyProcessRestaurant import next_occupancy_levels

# ── System parameters ──────────────────────────────────────────────────────────
_raw = get_fixed_data()
_d = {
    'P_max'    : _raw['heating_max_power'],
    'P_vent'   : _raw['ventilation_power'],
    'T_low'    : _raw['temp_min_comfort_threshold'],
    'T_OK'     : _raw['temp_OK_threshold'],
    'T_high'   : _raw['temp_max_comfort_threshold'],
    'H_high'   : _raw['humidity_threshold'],
    'zeta_exch': _raw['heat_exchange_coeff'],
    'zeta_loss': _raw['thermal_loss_coeff'],
    'zeta_conv': _raw['heating_efficiency_coeff'],
    'zeta_cool': _raw['heat_vent_coeff'],
    'zeta_occ' : _raw['heat_occupancy_coeff'],
    'eta_occ'  : _raw['humidity_occupancy_coeff'],
    'eta_vent' : _raw['humidity_vent_coeff'],
    'T_out'    : list(_raw['outdoor_temperature']),
}

# ── Load trained VFA weights ───────────────────────────────────────────────────
_weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'api_vfa_weights.json')
with open(_weights_path) as _f:
    _raw_json = json.load(_f)

# Convert string keys ("0".."9") to int
VFA_WEIGHTS = {int(t): w for t, w in _raw_json.items()}

# ── Number of scenarios for look-ahead (matches training) ─────────────────────
K_SCENARIOS = 15


# ── Single-step MILP with VFA look-ahead ──────────────────────────────────────
def _solve(state, next_weights):
    t      = int(state['current_time'])
    T_out  = float(_d['T_out'][min(t, 9)])
    T1, T2 = float(state['T1']), float(state['T2'])
    H      = float(state['H'])
    occ1, occ2 = float(state['occ1']), float(state['occ2'])
    price  = float(state['price'])
    c      = int(state.get('c', 0))

    m = pyo.ConcreteModel()
    m.p1 = pyo.Var(bounds=(0.0, _d['P_max']))
    m.p2 = pyo.Var(bounds=(0.0, _d['P_max']))
    m.v  = pyo.Var(domain=pyo.Binary)

    # Overrule constraints
    if c > 0:                         m.v.fix(1)
    if H >= _d['H_high']:             m.v.fix(1)
    if state.get('y_low_1'):          m.p1.fix(_d['P_max'])
    if state.get('y_low_2'):          m.p2.fix(_d['P_max'])
    if state.get('y_high_1'):         m.p1.fix(0.0)
    if state.get('y_high_2'):         m.p2.fix(0.0)

    # Next physical state (deterministic given action)
    m.T1x = pyo.Var()
    m.T2x = pyo.Var()
    m.Hx  = pyo.Var()
    m.cT1 = pyo.Constraint(expr=m.T1x == T1 + _d['zeta_exch']*(T2-T1) + _d['zeta_loss']*(T_out-T1) + _d['zeta_conv']*m.p1 - _d['zeta_cool']*m.v + _d['zeta_occ']*occ1)
    m.cT2 = pyo.Constraint(expr=m.T2x == T2 + _d['zeta_exch']*(T1-T2) + _d['zeta_loss']*(T_out-T2) + _d['zeta_conv']*m.p2 - _d['zeta_cool']*m.v + _d['zeta_occ']*occ2)
    m.cH  = pyo.Constraint(expr=m.Hx  == H  + _d['eta_occ']*(occ1+occ2) - _d['eta_vent']*m.v)

    # Next vent counter (linear in v for 3-step semantics)
    vc_next = (2.0 * m.v) if c == 0 else float(max(0, c - 1))

    immediate_cost = price * (m.p1 + m.p2 + _d['P_vent'] * m.v)

    expected_future = 0.0
    if next_weights is not None:
        # Generate K_SCENARIOS stochastic futures (price, occ)
        scenarios = []
        for _ in range(K_SCENARIOS):
            sc_p         = price_model(state['price'], state['price_previous'])
            sc_o1, sc_o2 = next_occupancy_levels(state['occ1'], state['occ2'])
            scenarios.append({'price': sc_p, 'occ1': sc_o1, 'occ2': sc_o2})

        # State-dependent VFA terms (same for all scenarios — factor out)
        vfa_phys = (
              next_weights['T1'] * ((m.T1x - 22.0) / 8.0)
            + next_weights['T2'] * ((m.T2x - 22.0) / 8.0)
            + next_weights['H']  * ((m.Hx  - 40.0) / 40.0)
            + next_weights['c']  * (vc_next          / 3.0)
        )
        # Deterministic part of stochastic features
        vfa_prev_price = next_weights['price_previous'] * (price / 10.0)

        # Average over scenario-specific (stochastic) terms
        avg_stochastic = (1.0 / K_SCENARIOS) * sum(
              next_weights['price'] * (sc['price'] / 10.0)
            + next_weights['occ1']  * ((sc['occ1'] - 20.0) / 30.0)
            + next_weights['occ2']  * ((sc['occ2'] - 10.0) / 20.0)
            for sc in scenarios
        )

        expected_future = (
            next_weights['intercept']
            + vfa_phys
            + vfa_prev_price
            + avg_stochastic
        )

    m.obj = pyo.Objective(expr=immediate_cost + expected_future, sense=pyo.minimize)
    slv = pyo.SolverFactory('gurobi')
    slv.options['OutputFlag'] = 0
    res = slv.solve(m)

    ok = (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible)
    if res.solver.termination_condition in ok:
        return {
            'HeatPowerRoom1': float(pyo.value(m.p1)),
            'HeatPowerRoom2': float(pyo.value(m.p2)),
            'VentilationON' : int(round(float(pyo.value(m.v)))),
        }

    # Fallback on solver failure
    return {
        'HeatPowerRoom1': _d['P_max'] if state.get('y_low_1') else 0.0,
        'HeatPowerRoom2': _d['P_max'] if state.get('y_low_2') else 0.0,
        'VentilationON' : 1 if (c > 0 or H >= _d['H_high']) else 0,
    }


# ── Public interface ───────────────────────────────────────────────────────────
def select_action(state: dict) -> dict:
    """
    API policy with time-indexed Ridge VFA (10 features, 10 time steps).
    Loads weights from api_vfa_weights.json at import time.
    """
    try:
        t = int(state.get('current_time', 0))
        c = int(state.get('c', 0))
        H = float(state.get('H', 0.0))

        # At the last time step there is no future to plan for — use dummy action.
        if t >= 9:
            return {
                'HeatPowerRoom1': _d['P_max'] if state.get('y_low_1') else 0.0,
                'HeatPowerRoom2': _d['P_max'] if state.get('y_low_2') else 0.0,
                'VentilationON' : 1 if (c > 0 or H >= _d['H_high']) else 0,
            }

        next_weights = VFA_WEIGHTS.get(t + 1)
        return _solve(state, next_weights)
    except Exception:
        c = int(state.get('c', 0))
        H = float(state.get('H', 0.0))
        return {
            'HeatPowerRoom1': _d['P_max'] if state.get('y_low_1') else 0.0,
            'HeatPowerRoom2': _d['P_max'] if state.get('y_low_2') else 0.0,
            'VentilationON' : 1 if (c > 0 or H >= _d['H_high']) else 0,
        }
