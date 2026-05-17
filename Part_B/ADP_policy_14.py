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
# 2. SCALED BASIS FUNCTION
#
# Scaling constants (5, 100, 2, 4) must match the notebook (cell-03 / cell-06).
# ==============================================================================
def _phi(state: dict) -> np.ndarray:
    """Scaled 7-feature basis function — all features ≈ O(1)."""
    T1 = float(state['T1'])
    T2 = float(state['T2'])
    H  = float(state.get('H', 0.0))
    c  = float(state.get('c', 0))
    T_ok  = params['T_ok']
    T_low = params['T_low']
    return np.array([
        ((T1 - T_ok) / 5.0)**2,
        ((T2 - T_ok) / 5.0)**2,
        H / 100.0,
        c / 2.0,
        max(0.0, T_low - T1) / 4.0,
        max(0.0, T_low - T2) / 4.0,
        1.0,
    ])


# ==============================================================================
# 3. TIME-DEPENDENT VFA WEIGHTS  (produced by ADP_policy_14.ipynb)
#
# ETA[t] is a 7-element list corresponding to the scaled features in _phi().
# ==============================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS_PATH = os.path.join(_HERE, 'output', 'adp_weights.json')

try:
    with open(_WEIGHTS_PATH) as _f:
        _w = json.load(_f)
    ETA = {int(t): _w['eta'][t] for t in _w['eta']}
except FileNotFoundError:
    # Fallback weights — reasonable order-of-magnitude values on scaled features
    _fallback = [10.0, 10.0, 5.0, 3.0, 8.0, 8.0, 0.0]
    ETA = {t: _fallback for t in range(10)}

# ==============================================================================
# 4. SCENARIO GENERATION PARAMETERS
# ==============================================================================
K = 5                          # number of next-period scenarios
_OCC1_LO, _OCC1_HI = 25, 35   # occupancy range room 1 (persons)
_OCC2_LO, _OCC2_HI = 15, 25   # occupancy range room 2 (persons)
_PRICE_LO, _PRICE_HI = 2, 8   # electricity price range (euro/kWh)

_rng = np.random.default_rng(seed=None)  # seeded fresh each import


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
                       + (1/K) * sum_{k=1}^K [ eta_{t+1}^T * phi_scaled(x_{k,t+1}) ]

    Scaling inside the VFA (factors 5, 100, 2, 4) matches _phi() and the
    notebook's phi() / solve_1step().
    """
    p     = params
    P_max = p['P_max']
    T_ok  = p['T_ok']
    T_low = p['T_low']
    t     = int(state.get('current_time', 0))
    T_out = float(p['T_out'][min(t, 9)])
    T1    = float(state['T1'])
    T2    = float(state['T2'])
    H     = float(state.get('H', 0.0))
    c     = int(state.get('c', 0))
    price = float(state['price'])

    occ1_k, occ2_k, _ = _sample_scenarios()   # K next-period occ samples

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
    m.pen1 = pyo.Var(m.K_set, bounds=(0, 20.0))
    m.pen2 = pyo.Var(m.K_set, bounds=(0, 20.0))

    # ── Per-scenario dynamics constraints ─────────────────────────────────────
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

    def _pen1_c(m, k):
        return m.pen1[k] >= T_low - m.T1x[k]

    def _pen2_c(m, k):
        return m.pen2[k] >= T_low - m.T2x[k]

    m.dyn_T1 = pyo.Constraint(m.K_set, rule=_dT1)
    m.dyn_T2 = pyo.Constraint(m.K_set, rule=_dT2)
    m.dyn_H  = pyo.Constraint(m.K_set, rule=_dH)
    m.pen1_c = pyo.Constraint(m.K_set, rule=_pen1_c)
    m.pen2_c = pyo.Constraint(m.K_set, rule=_pen2_c)

    # ── Overrule constraints (hard — state flags are fully observed online) ───
    if c > 0:
        m.vc = pyo.Constraint(expr=m.v == 1)
    if H >= p['H_high']:
        m.hc = pyo.Constraint(expr=m.v == 1)
    if state.get('y_low_1'):
        m.h1l = pyo.Constraint(expr=m.p1 == P_max)
    if state.get('y_low_2'):
        m.h2l = pyo.Constraint(expr=m.p2 == P_max)
    if state.get('y_high_1'):
        m.h1h = pyo.Constraint(expr=m.p1 == 0.0)
    if state.get('y_high_2'):
        m.h2h = pyo.Constraint(expr=m.p2 == 0.0)

    # c_next is linear in v given current c
    # c=0 → 2v  |  c=1, v forced=1 → 0  |  c=2, v forced=1 → 1
    if   c == 0: c_next = 2.0 * m.v
    elif c == 1: c_next = 0.0
    else:        c_next = 1.0

    # ── K-scenario VFA with scaled features ──────────────────────────────────
    eta_k_sum = sum(
        eta[0]*((m.T1x[k] - T_ok) / 5.0)**2 +
        eta[1]*((m.T2x[k] - T_ok) / 5.0)**2 +
        eta[2]*(m.Hx[k] / 100.0) +
        eta[3]*(c_next / 2.0) +
        eta[4]*(m.pen1[k] / 4.0) +
        eta[5]*(m.pen2[k] / 4.0) +
        eta[6]
        for k in range(K)
    )
    vfa_expected = eta_k_sum / K

    m.obj = pyo.Objective(
        expr=price*(m.p1 + m.p2 + p['P_vent']*m.v) + vfa_expected,
        sense=pyo.minimize)

    solver = pyo.SolverFactory('gurobi')
    solver.options['OutputFlag'] = 0
    solver.options['NonConvex']  = 2

    result = solver.solve(m)

    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
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

    # ── Diagnostics: compare immediate cost vs VFA contribution ──────────────
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