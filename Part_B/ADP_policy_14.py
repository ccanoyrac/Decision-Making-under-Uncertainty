import os
import json
import numpy as np
import pyomo.environ as pyo
import v2_SystemCharacteristics as sc

# ==============================================================================
# 1. SYSTEM PARAMETERS
# ==============================================================================
_raw = sc.get_fixed_data()
params = {
    'P_max'    : _raw['heating_max_power'],
    'P_vent'   : _raw['ventilation_power'],
    'T_low'    : _raw['temp_min_comfort_threshold'],
    'T_ok'     : _raw['temp_OK_threshold'],
    'T_high'   : _raw['temp_max_comfort_threshold'],
    'H_high'   : _raw['humidity_threshold'],
    'zeta_exch': _raw['heat_exchange_coeff'],
    'zeta_loss': _raw['thermal_loss_coeff'],
    'zeta_conv': _raw['heating_efficiency_coeff'],
    'zeta_cool': _raw['heat_vent_coeff'],
    'zeta_occ' : _raw['heat_occupancy_coeff'],
    'eta_occ'  : _raw['humidity_occupancy_coeff'],
    'eta_vent' : _raw['humidity_vent_coeff'],
    'T_out'    : _raw['outdoor_temperature'],
}

# ==============================================================================
# 2. SCALED BASIS FUNCTION  (must match notebook cell-03 and cell-06)
#
# ETA[t] is a 7-element list for features:
#   [((T1-T_ok)/5)^2,  ((T2-T_ok)/5)^2,  H/100,  c/2,
#    max(0,T_warn-T1)/5,  max(0,T_warn-T2)/5,  1]
# where T_warn = T_low + 1.0 = 19°C.
# ==============================================================================
_T_WARN = _raw['temp_min_comfort_threshold'] + 1.0   # 19°C early-warning threshold


def _phi(state: dict) -> np.ndarray:
    """Scaled 7-feature basis function — all features ≈ O(1)."""
    T1 = float(state['T1'])
    T2 = float(state['T2'])
    H  = float(state.get('H', 0.0))
    c  = float(state.get('c', 0))
    T_ok = params['T_ok']
    return np.array([
        ((T1 - T_ok) / 5.0)**2,
        ((T2 - T_ok) / 5.0)**2,
        H / 100.0,
        c / 2.0,
        max(0.0, _T_WARN - T1) / 5.0,
        max(0.0, _T_WARN - T2) / 5.0,
        1.0,
    ])


# ==============================================================================
# 3. TIME-DEPENDENT VFA WEIGHTS  (produced by ADP_policy_14.ipynb)
# ==============================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS_PATH = os.path.join(_HERE, 'output', 'adp_weights.json')

try:
    with open(_WEIGHTS_PATH) as _f:
        _w = json.load(_f)
    ETA = {int(t): _w['eta'][t] for t in _w['eta']}
    # Sanitize in-place: replace any NaN/Inf with 0.0 so Pyomo never sees bad floats
    _n_fixed = 0
    for _t, _v in ETA.items():
        for _i, _val in enumerate(_v):
            if not np.isfinite(_val):
                _v[_i] = 0.0
                _n_fixed += 1
    if _n_fixed:
        print(f"[ADP] WARNING: Replaced {_n_fixed} NaN/Inf weight values with 0.0")
    print(f"[ADP] Loaded weights OK — {len(ETA)} timesteps")
    for _t in sorted(ETA)[:3]:
        print(f"[ADP]   t={_t}: {np.round(ETA[_t], 4)}")
except FileNotFoundError:
    print(f"[ADP] WARNING: {_WEIGHTS_PATH} not found — using fallback weights")
    _fallback = [10.0, 10.0, 5.0, 3.0, 8.0, 8.0, 0.0]
    ETA = {t: _fallback for t in range(10)}

# ==============================================================================
# 4. SCENARIO GENERATION PARAMETERS
# ==============================================================================
K = 50                         # number of next-period scenarios
_OCC1_LO, _OCC1_HI = 25, 35   # occupancy range room 1 (persons)
_OCC2_LO, _OCC2_HI = 15, 25   # occupancy range room 2 (persons)
_PRICE_LO, _PRICE_HI = 2, 8   # electricity price range (euro/kWh)

_rng = np.random.default_rng(seed=None)


def _sample_scenarios():
    """Draw K independent scenarios for next-period occupancy and price."""
    occ1_k  = _rng.uniform(_OCC1_LO,  _OCC1_HI,  size=K).tolist()
    occ2_k  = _rng.uniform(_OCC2_LO,  _OCC2_HI,  size=K).tolist()
    price_k = _rng.uniform(_PRICE_LO, _PRICE_HI, size=K).tolist()
    return occ1_k, occ2_k, price_k


# ==============================================================================
# 5. ONLINE K-SCENARIO ADP OPTIMIZER
# ==============================================================================
def solve_adp_step(state: dict, eta: list, params: dict) -> dict:
    """
    Solve the here-and-now MIQP with K-scenario expectation:

        min_{p1,p2,v}  price*(p1 + p2 + P_vent*v)
                       + (1/K) * sum_k [ eta^T * phi_scaled(x_{k,t+1}) ]

    pen1[k]/pen2[k] linearise max(0, T_warn - T_rx[k]) with T_warn = 19°C.
    All /5.0 scalings match _phi() and the notebook's phi() / solve_1step().
    """
    p      = params
    P_max  = p['P_max']
    T_ok   = p['T_ok']
    T_warn = _T_WARN               # 19°C — must match _phi()
    t      = int(state.get('current_time', 0))
    T_out  = float(p['T_out'][min(t, 9)])
    T1     = float(state['T1'])
    T2     = float(state['T2'])
    H      = float(state.get('H', 0.0))
    c      = int(state.get('c', 0))
    price  = float(state['price'])

    occ1_k, occ2_k, _ = _sample_scenarios()

    m = pyo.ConcreteModel()

    # ── Shared decision variables ─────────────────────────────────────────────
    m.p1 = pyo.Var(bounds=(0, P_max))
    m.p2 = pyo.Var(bounds=(0, P_max))
    m.v  = pyo.Var(domain=pyo.Binary)

    # ── Per-scenario next-state variables ─────────────────────────────────────
    m.K_set = pyo.RangeSet(0, K - 1)
    m.T1x  = pyo.Var(m.K_set)
    m.T2x  = pyo.Var(m.K_set)
    m.Hx   = pyo.Var(m.K_set)
    m.pen1 = pyo.Var(m.K_set, domain=pyo.NonNegativeReals)  # unbounded: extreme scenarios never infeasible
    m.pen2 = pyo.Var(m.K_set, domain=pyo.NonNegativeReals)

    # ── Per-scenario dynamics ─────────────────────────────────────────────────
    def _dT1(m, k):
        return m.T1x[k] == (T1 + p['zeta_exch']*(T2 - T1)
                            + p['zeta_loss']*(T_out - T1)
                            + p['zeta_conv']*m.p1
                            - p['zeta_cool']*m.v
                            + p['zeta_occ']*occ1_k[k])

    def _dT2(m, k):
        return m.T2x[k] == (T2 + p['zeta_exch']*(T1 - T2)
                            + p['zeta_loss']*(T_out - T2)
                            + p['zeta_conv']*m.p2
                            - p['zeta_cool']*m.v
                            + p['zeta_occ']*occ2_k[k])

    def _dH(m, k):
        return m.Hx[k] == (H + p['eta_occ']*(occ1_k[k] + occ2_k[k])
                           - p['eta_vent']*m.v)

    # Early-warning pen: triggers at T_warn = 19°C (1°C before hard overrule)
    def _pen1_c(m, k):
        return m.pen1[k] >= T_warn - m.T1x[k]

    def _pen2_c(m, k):
        return m.pen2[k] >= T_warn - m.T2x[k]

    m.dyn_T1 = pyo.Constraint(m.K_set, rule=_dT1)
    m.dyn_T2 = pyo.Constraint(m.K_set, rule=_dT2)
    m.dyn_H  = pyo.Constraint(m.K_set, rule=_dH)
    m.pen1_c = pyo.Constraint(m.K_set, rule=_pen1_c)
    m.pen2_c = pyo.Constraint(m.K_set, rule=_pen2_c)

    # ── Big-M soft overrule penalties (same pattern as notebook; always feasible) ──
    _penalty = []
    if c > 0 or H >= p['H_high']:
        _penalty.append(1_000 * (1 - m.v))
    if state.get('y_low_1'):
        _penalty.append(1_000 * (P_max - m.p1))
    if state.get('y_low_2'):
        _penalty.append(1_000 * (P_max - m.p2))
    if state.get('y_high_1'):
        _penalty.append(1_000 * m.p1)
    if state.get('y_high_2'):
        _penalty.append(1_000 * m.p2)
    overrule_penalty = sum(_penalty) if _penalty else 0.0

    # c_next: c=0 → 2v | c=1 → 0 (v irrelevant) | c=2 → 1 (Big-M drives v=1)
    if   c == 0: c_next = 2.0 * m.v
    elif c == 1: c_next = 0.0
    else:        c_next = 1.0

    # ── K-scenario VFA with scaled features ──────────────────────────────────
    eta_k_sum = sum(
        eta[0]*((m.T1x[k] - T_ok) / 5.0)**2 +
        eta[1]*((m.T2x[k] - T_ok) / 5.0)**2 +
        eta[2]*(m.Hx[k] / 100.0) +
        eta[3]*(c_next / 2.0) +
        eta[4]*(m.pen1[k] / 5.0) +
        eta[5]*(m.pen2[k] / 5.0) +
        eta[6]
        for k in range(K)
    )
    vfa_expected = eta_k_sum / K

    m.obj = pyo.Objective(
        expr=price*(m.p1 + m.p2 + p['P_vent']*m.v) + vfa_expected + overrule_penalty,
        sense=pyo.minimize)

    solver = pyo.SolverFactory('gurobi')
    solver.options['OutputFlag'] = 0
    solver.options['NonConvex']  = 2

    result = solver.solve(m)

    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        print(f"SOLVER FAILED at t={t}: {result.solver.termination_condition}")
        return {'HeatPowerRoom1': 0.0, 'HeatPowerRoom2': 0.0, 'VentilationON': 0}

    return {
        'HeatPowerRoom1': float(pyo.value(m.p1)),
        'HeatPowerRoom2': float(pyo.value(m.p2)),
        'VentilationON' : int(round(pyo.value(m.v))),
    }


# ==============================================================================
# 6. POLICY ENTRY POINT  (called by Task6_Environment.run_policy)
# ==============================================================================
def select_action(state: dict) -> dict:
    t   = int(state.get('current_time', 0))
    eta = ETA.get(t, ETA[max(ETA.keys())])
    decisions = solve_adp_step(state, eta, params)

    # ── Diagnostics: immediate cost vs VFA at current state ──────────────────
    p1    = decisions['HeatPowerRoom1']
    p2    = decisions['HeatPowerRoom2']
    v     = decisions['VentilationON']
    price = float(state['price'])
    imm   = price * (p1 + p2 + params['P_vent'] * v)
    vfa   = float(np.dot(eta, _phi(state)))
    print(f"  t={t} | p1={p1:.2f} p2={p2:.2f} v={v}"
          f" | imm={imm:.3f} vfa_curr={vfa:.2f} | price={price:.2f}"
          f" | T1={state['T1']:.1f} T2={state['T2']:.1f}")

    return decisions
