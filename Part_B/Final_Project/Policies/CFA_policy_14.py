"""
CFA_policy_14.py
================
Pure Cost Function Approximation (CFA) policy — Group 14, DTU 02435, Spring 2026.

Architecture
------------
Single-step MILP: minimise the immediate energy cost plus a linear VFA terminal
cost trained offline by ADP_Training_NEW.py (weights in api_vfa_weights.json).

    min  price * (p1 + p2 + P_vent * v)          [immediate cost]
       + VFA_{t+1}(T1_next, T2_next, H_next, ...)  [CFA terminal cost]

No scenario tree is built — this is the CFA-only component of Hybrid_policy.py.
Uncertainty beyond the current step is handled entirely by the VFA.

VFA features (same normalisation as Hybrid_policy.py):
    T1_feat        = (T1_next - 22) / 8
    T2_feat        = (T2_next - 22) / 8
    H_feat         = (H_next  - 40) / 40
    c_feat         ≈ v * (2/3)                  [linear approx, keeps MILP linear]
    price_prev_feat = price_t / 10              [current price becomes prev next step]
    E[price_next, occ1_next, occ2_next]         [MC draws from stochastic process models]

Public interface:
    select_action(state) -> {'HeatPowerRoom1', 'HeatPowerRoom2', 'VentilationON'}
"""

import os
import json
import numpy as np
import pyomo.environ as pyo
from  Data.v2_SystemCharacteristics import get_fixed_data
from  Data.PriceProcessRestaurant    import price_model
from  Data.OccupancyProcessRestaurant import next_occupancy_levels

# ── System parameters ─────────────────────────────────────────────────────────
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
_T_OUT = _SYS['outdoor_temperature']

# ── VFA weights (trained by ADP_Training_NEW.py) ──────────────────────────────
_dir      = os.path.dirname(os.path.abspath(__file__))
_VFA_PATH = os.path.join(_dir, 'api_vfa_weights.json')
try:
    with open(_VFA_PATH) as _f:
        _VFA = {int(k): v for k, v in json.load(_f).items()}
except FileNotFoundError:
    _VFA = {}

def _t_out(t: int) -> float:
    return float(_T_OUT[max(0, min(t, len(_T_OUT) - 1))])


def _sample_price_proc(price_t: float, price_prev: float, n: int) -> np.ndarray:
    return np.array([price_model(price_t, price_prev) for _ in range(n)])


def _sample_occ_proc(occ1: float, occ2: float, n: int):
    samples = [next_occupancy_levels(occ1, occ2) for _ in range(n)]
    return (np.array([s[0] for s in samples]),
            np.array([s[1] for s in samples]))


def select_action(state: dict) -> dict:
    """
    CFA-only policy: single-step MILP + VFA terminal cost (no scenario tree).

    Parameters
    ----------
    state : dict
        Keys: T1, T2, H, Occ1, Occ2, price_t, price_previous,
              vent_counter, low_override_r1, low_override_r2, current_time.

    Returns
    -------
    dict with keys HeatPowerRoom1, HeatPowerRoom2, VentilationON.
    """
    K_STOCH = 40   # MC draws for stochastic VFA terms (price, occ at t+1)

    p  = _P
    t  = int(state.get('current_time', 0))
    c  = int(state.get('vent_counter', 0))
    H  = float(state.get('H', 0.0))

    y_lo1_0 = int(state.get('low_override_r1', 0))
    y_lo2_0 = int(state.get('low_override_r2', 0))

    if t >= 9:
        return {
            'HeatPowerRoom1': p['P_max'] if y_lo1_0 else 0.0,
            'HeatPowerRoom2': p['P_max'] if y_lo2_0 else 0.0,
            'VentilationON' : 1 if (c > 0 or H >= p['H_high']) else 0,
        }

    T1_obs = float(state['T1'])
    T2_obs = float(state['T2'])
    H_obs  = float(state['H'])
    occ1        = float(state['Occ1'])
    occ2        = float(state['Occ2'])
    price       = float(state['price_t'])
    price_prev  = float(state.get('price_previous', price))
    To          = _t_out(t)

    y_hi1_0 = 1 if T1_obs > p['T_high'] else 0
    y_hi2_0 = 1 if T2_obs > p['T_high'] else 0
    h_ov_0  = 1 if H_obs  > p['H_high'] else 0

    w = _VFA.get(t + 1, {})

    try:
        m = pyo.ConcreteModel()

        # ── Decision variables ─────────────────────────────────────────────────
        m.p1 = pyo.Var(bounds=(0.0, p['P_max']))
        m.p2 = pyo.Var(bounds=(0.0, p['P_max']))
        m.v  = pyo.Var(domain=pyo.Binary)

        # ── Overrule hard constraints ──────────────────────────────────────────
        if c > 0 or h_ov_0: m.v.fix(1)
        if y_hi1_0:          m.p1.fix(0.0)
        if y_hi2_0:          m.p2.fix(0.0)
        if y_lo1_0:          m.p1.fix(p['P_max'])
        if y_lo2_0:          m.p2.fix(p['P_max'])

        # ── Next physical state (linear in decision variables) ─────────────────
        m.T1x = pyo.Var()
        m.T2x = pyo.Var()
        m.Hx  = pyo.Var()
        m.cT1 = pyo.Constraint(expr=m.T1x == T1_obs
                                + p['zeta_exch'] * (T2_obs - T1_obs)
                                + p['zeta_loss'] * (To - T1_obs)
                                + p['zeta_conv'] * m.p1
                                - p['zeta_cool'] * m.v
                                + p['zeta_occ']  * occ1)
        m.cT2 = pyo.Constraint(expr=m.T2x == T2_obs
                                + p['zeta_exch'] * (T1_obs - T2_obs)
                                + p['zeta_loss'] * (To - T2_obs)
                                + p['zeta_conv'] * m.p2
                                - p['zeta_cool'] * m.v
                                + p['zeta_occ']  * occ2)
        m.cH  = pyo.Constraint(expr=m.Hx == H_obs
                                + p['eta_occ']  * (occ1 + occ2)
                                - p['eta_vent'] * m.v)

        # ── Immediate cost ─────────────────────────────────────────────────────
        immediate = price * (m.p1 + m.p2 + p['P_vent'] * m.v)

        # ── VFA terminal cost ──────────────────────────────────────────────────
        vfa_expr = 0.0
        if w:
            T1_feat = (m.T1x - 22.0) / 8.0
            T2_feat = (m.T2x - 22.0) / 8.0
            H_feat  = (m.Hx  - 40.0) / 40.0

            # c_feat: linear approximation of vent_counter after this step
            # (keeps the MILP LP-relaxation linear, same as Hybrid_policy.py)
            if c >= 2:
                c_feat = float(max(0, c - 2)) / 3.0
            else:
                c_feat = m.v * (2.0 / 3.0)

            price_prev_feat = price / 10.0

            det_expr = (
                  w['T1']            * T1_feat
                + w['T2']            * T2_feat
                + w['H']             * H_feat
                + w['c']             * c_feat
                + w['price_previous'] * price_prev_feat
            )

            # Stochastic terms: MC draws for price and occupancy at t+1
            p_smp          = _sample_price_proc(price, price_prev, K_STOCH)
            o1_smp, o2_smp = _sample_occ_proc(occ1, occ2, K_STOCH)
            stoch_const = float(np.mean(
                  w['price'] * p_smp / 10.0
                + w['occ1']  * (o1_smp - 20.0) / 30.0
                + w['occ2']  * (o2_smp - 10.0) / 20.0
            ))

            vfa_expr = det_expr + stoch_const + w['intercept']

        m.obj = pyo.Objective(expr=immediate + vfa_expr, sense=pyo.minimize)

        solver = pyo.SolverFactory('gurobi')
        solver.options['OutputFlag'] = 0
        solver.options['TimeLimit']  = 5
        res = solver.solve(m)

        ok = (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible)
        if res.solver.termination_condition in ok:
            return {
                'HeatPowerRoom1': float(pyo.value(m.p1)),
                'HeatPowerRoom2': float(pyo.value(m.p2)),
                'VentilationON' : int(round(float(pyo.value(m.v)))),
            }

    except Exception:
        pass

    # Fallback: reactive
    return {
        'HeatPowerRoom1': p['P_max'] if y_lo1_0 else 0.0,
        'HeatPowerRoom2': p['P_max'] if y_lo2_0 else 0.0,
        'VentilationON' : 1 if (c > 0 or H_obs > p['H_high']) else 0,
    }
