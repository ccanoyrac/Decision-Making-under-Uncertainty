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
# 2. TIME-DEPENDENT VFA WEIGHTS  (produced by ADP_policy_14.ipynb)
#
# ETA[t] is a 7-element list corresponding to features:
#   [(T1-T_ok)^2, (T2-T_ok)^2, H, c, max(0,T_low-T1), max(0,T_low-T2), 1]
# ==============================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS_PATH = os.path.join(_HERE, 'output', 'adp_weights.json')

try:
    with open(_WEIGHTS_PATH) as _f:
        _w = json.load(_f)
    ETA = {int(t): _w['eta'][t] for t in _w['eta']}
except FileNotFoundError:
    # Fallback weights (kept for safety before first notebook run)
    _fallback = [2.2667, 2.2761, 5.1160, 0.0270, 0.0368, 0.0, 0.0]
    ETA = {t: _fallback for t in range(10)}

# ==============================================================================
# 3. SCENARIO GENERATION PARAMETERS
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
# 4. ONLINE K-SCENARIO ADP OPTIMIZER
# ==============================================================================
def solve_adp_step(state: dict, eta: list, params: dict) -> dict:
    """
    Solve the here-and-now MIQP with K-scenario expectation:

        min_{p1,p2,v}  price*(p1 + p2 + P_vent*v)
                       + (1/K) * sum_{k=1}^K [ eta_{t+1}^T * phi(x_{k,t+1}(p1,p2,v)) ]

    For each scenario k a different occupancy sample (occ1_k, occ2_k) is drawn
    from the next-period distribution, giving K different post-decision states
    x_{k,t+1}.  The decision variables (p1, p2, v) are shared across scenarios.

    phi(x_next) = [(T1x-T_ok)^2, (T2x-T_ok)^2, Hx, c_next, pen1, pen2, 1]
    expressed as Pyomo expressions via the post-decision dynamics.
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

    # ── K-scenario VFA: average eta^T * phi(x_{k,t+1}) over all scenarios ────
    eta_k_sum = sum(
        eta[0]*(m.T1x[k] - T_ok)**2 +
        eta[1]*(m.T2x[k] - T_ok)**2 +
        eta[2]*m.Hx[k] +
        eta[3]*c_next +
        eta[4]*m.pen1[k] +
        eta[5]*m.pen2[k] +
        eta[6]
        for k in range(K)
    )
    vfa_expected = eta_k_sum / K

    m.obj = pyo.Objective(
        expr=price*(m.p1 + m.p2 + p['P_vent']*m.v) + vfa_expected,
        sense=pyo.minimize)

    solver = pyo.SolverFactory('gurobi')
    solver.options['OutputFlag'] = 0
    solver.options['NonConvex']  = 2   # required for quadratic VFA terms

    result = solver.solve(m)

    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        return {'HeatPowerRoom1': 0.0, 'HeatPowerRoom2': 0.0, 'VentilationON': 0}

    return {
        'HeatPowerRoom1': float(pyo.value(m.p1)),
        'HeatPowerRoom2': float(pyo.value(m.p2)),
        'VentilationON' : int(round(pyo.value(m.v))),
    }


# ==============================================================================
# 5. POLICY ENTRY POINT  (called by Task6_Environment.run_policy)
# ==============================================================================
def select_action(state: dict) -> dict:
    t   = int(state.get('current_time', 0))
    eta = ETA.get(t, ETA[max(ETA.keys())])
    return solve_adp_step(state, eta, params)