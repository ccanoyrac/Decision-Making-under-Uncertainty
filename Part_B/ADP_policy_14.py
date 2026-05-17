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
# 2. LINEAR BASIS FUNCTION  (must match notebook cell-03)
#
# phi(state) — 9 features, all O(1):
#   [T1/30, T2/30, H/100, price/10, price_prev/10, occ1/40, occ2/30, c/2, 1]
# ==============================================================================
N_FEAT = 11


def _phi(state: dict) -> np.ndarray:
    T1 = float(state['T1'])
    T2 = float(state['T2'])
    return np.array([
        T1                                                 / 30.0,
        T2                                                 / 30.0,
        float(state.get('H', 0.0))                         / 100.0,
        float(state['price'])                              / 10.0,
        float(state.get('price_previous', state['price'])) / 10.0,
        float(state.get('occ1', 0.0))                      / 40.0,
        float(state.get('occ2', 0.0))                      / 30.0,
        float(state.get('c', 0))                           / 2.0,
        max(0.0, 19.5 - T1)                                / 5.0,
        max(0.0, 19.5 - T2)                                / 5.0,
        1.0,
    ])


# ==============================================================================
# 3. LINEAR VFA WEIGHTS  (produced by ADP_policy_14.ipynb)
# ==============================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS_PATH = os.path.join(_HERE, 'output', 'adp_weights.json')

try:
    with open(_WEIGHTS_PATH) as _f:
        _w = json.load(_f)
    ETA = {int(t): _w['eta'][t] for t in _w['eta']}
    # Sanitize: replace any NaN/Inf with 0.0 so Pyomo never sees bad floats
    _n_fixed = 0
    for _t, _v in ETA.items():
        for _i, _val in enumerate(_v):
            if not np.isfinite(_val):
                _v[_i] = 0.0
                _n_fixed += 1
    if _n_fixed:
        print(f"[ADP] WARNING: Replaced {_n_fixed} NaN/Inf weight values with 0.0")
    print(f"[ADP] Loaded {len(ETA)} timesteps × {len(next(iter(ETA.values())))} features")
    for _t in sorted(ETA)[:3]:
        print(f"[ADP]   t={_t}: {np.round(ETA[_t], 4)}")
except FileNotFoundError:
    print(f"[ADP] WARNING: {_WEIGHTS_PATH} not found — using zero weights")
    ETA = {t: [0.0] * N_FEAT for t in range(10)}  # N_FEAT=11

# ==============================================================================
# 4. EXPECTED NEXT-PERIOD EXOGENOUS VALUES  (K=50 Monte-Carlo mean)
#
# Because the VFA is strictly linear:  E[V(x)] = V(E[x])
# So sampling K scenarios and averaging their mean is equivalent to using the
# true expected value directly — a single scalar per variable suffices.
# ==============================================================================
K = 50
_OCC1_LO,  _OCC1_HI  = 25, 35
_OCC2_LO,  _OCC2_HI  = 15, 25
_PRICE_LO, _PRICE_HI =  2,  8

_rng = np.random.default_rng(seed=None)


def _expected_exogenous():
    """Sample K next-period scenarios and return their means."""
    return (
        float(np.mean(_rng.uniform(_OCC1_LO,  _OCC1_HI,  K))),
        float(np.mean(_rng.uniform(_OCC2_LO,  _OCC2_HI,  K))),
        float(np.mean(_rng.uniform(_PRICE_LO, _PRICE_HI, K))),
    )


# ==============================================================================
# 5. 1-STEP LINEAR MILP
# ==============================================================================
def solve_adp_step(state: dict, eta: list, params: dict) -> dict:
    """
    Solve the 1-step MILP using expected next-period exogenous variables.

    E[V(x_{t+1})] = V(E[x_{t+1}]) holds because the VFA is linear and the
    dynamics are linear in the exogenous variables.  A single expected scenario
    replaces the K-scenario sum — the model is a standard MILP with no quadratic
    terms and no NonConvex flag.

    Objective:
        min_{p1,p2,v}  price*(p1 + p2 + P_vent*v)
                       + eta_{t+1}^T * phi(E[x_{t+1}])
                       + Big-M overrule penalties
    """
    p     = params
    P_max = p['P_max']
    t     = int(state.get('current_time', 0))
    T_out = float(p['T_out'][min(t, 9)])
    T1    = float(state['T1'])
    T2    = float(state['T2'])
    H     = float(state.get('H', 0.0))
    c     = int(state.get('c', 0))
    price = float(state['price'])

    exp_occ1, exp_occ2, exp_price_next = _expected_exogenous()

    m = pyo.ConcreteModel()

    # ── Decision variables ─────────────────────────────────────────────────────
    m.p1  = pyo.Var(bounds=(0, P_max))
    m.p2  = pyo.Var(bounds=(0, P_max))
    m.v   = pyo.Var(domain=pyo.Binary)

    # ── Expected next-state physical variables (linear in decisions) ───────────
    m.T1x = pyo.Var()
    m.T2x = pyo.Var()
    m.Hx  = pyo.Var()

    # ── Dynamics (realized current occ for physical transition) ────────────────
    m.dT1 = pyo.Constraint(expr=
        m.T1x == T1 + p['zeta_exch']*(T2 - T1)
               + p['zeta_loss']*(T_out - T1)
               + p['zeta_conv']*m.p1
               - p['zeta_cool']*m.v
               + p['zeta_occ']*exp_occ1)
    m.dT2 = pyo.Constraint(expr=
        m.T2x == T2 + p['zeta_exch']*(T1 - T2)
               + p['zeta_loss']*(T_out - T2)
               + p['zeta_conv']*m.p2
               - p['zeta_cool']*m.v
               + p['zeta_occ']*exp_occ2)
    m.dH  = pyo.Constraint(expr=
        m.Hx  == H + p['eta_occ']*(exp_occ1 + exp_occ2)
               - p['eta_vent']*m.v)

    # ── c_next (ventilation inertia) ───────────────────────────────────────────
    if   c == 0: c_next = 2.0 * m.v
    elif c == 1: c_next = 0.0
    else:        c_next = 1.0

    # ── LP linearisation of pen features for next state ───────────────────────
    # pen_rx = max(0, 19.5 - T_rx) / 5.  With eta[8]/eta[9] > 0, minimisation
    # drives pen_rx to its lower bound = max(0, (19.5-T_rx)/5).
    m.pen1x   = pyo.Var(bounds=(0, 5.0))
    m.pen2x   = pyo.Var(bounds=(0, 5.0))
    m.c_pen1x = pyo.Constraint(expr=m.pen1x >= (19.5 - m.T1x) / 5.0)
    m.c_pen2x = pyo.Constraint(expr=m.pen2x >= (19.5 - m.T2x) / 5.0)

    # ── Linear VFA of expected next state (11 features) ───────────────────────
    # phi = [T1x/30, T2x/30, Hx/100, E[price]/10, price_t/10,
    #        E[occ1]/40, E[occ2]/30, c_next/2, pen1x, pen2x, 1]
    vfa_next = (eta[0] *(m.T1x         / 30.0) +
                eta[1] *(m.T2x         / 30.0) +
                eta[2] *(m.Hx          / 100.0) +
                eta[3] *(exp_price_next / 10.0) +
                eta[4] *(price          / 10.0) +
                eta[5] *(exp_occ1       / 40.0) +
                eta[6] *(exp_occ2       / 30.0) +
                eta[7] *(c_next         / 2.0) +
                eta[8] * m.pen1x +
                eta[9] * m.pen2x +
                eta[10])

    # ── Big-M soft overrule penalties ─────────────────────────────────────────
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

    m.obj = pyo.Objective(
        expr=price*(m.p1 + m.p2 + p['P_vent']*m.v) + vfa_next + overrule_penalty,
        sense=pyo.minimize)

    solver = pyo.SolverFactory('gurobi')
    solver.options['OutputFlag'] = 0

    try:
        result = solver.solve(m)
    except Exception as e:
        print(f"PYOMO CRASHED: {e}")
        return {'HeatPowerRoom1': 0.0, 'HeatPowerRoom2': 0.0, 'VentilationON': 0}

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
    t         = int(state.get('current_time', 0))
    eta       = ETA.get(t, ETA[max(ETA.keys())])
    decisions = solve_adp_step(state, eta, params)

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