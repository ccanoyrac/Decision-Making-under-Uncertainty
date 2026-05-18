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
# 2. FEATURE MAPPING  (must match the normalisation used during ADP training)
#
# phi(x_t) in R^12 — normalised to a stable range:
#
#   idx  feature          formula                              baseline  scale
#    0   T1               (T1   - 22) / 8                     22 °C     8 °C
#    1   T2               (T2   - 22) / 8                     22 °C     8 °C
#    2   H                (H    - 40) / 40                     40 %      40 %
#    3   price             price / 10                          0         10 €/kWh
#    4   price_previous    price_prev / 10                     0         10 €/kWh
#    5   occ1             (occ1 - 30) / 30                     30 pax    30
#    6   occ2             (occ2 - 20) / 20                     20 pax    20
#    7   c                 c / 3                               0         3
#    8   pen1              max(0, 22.0 - T1) / 3               —         3 °C
#    9   pen2              max(0, 22.0 - T2) / 3               —         3 °C
#   10   price × pen1      (price/10) * max(0,22.0-T1)/3       —         —
#   11   price × pen2      (price/10) * max(0,22.0-T2)/3       —         —
#
# Features 10/11 capture the interaction "cold rooms during high-price periods
# are disproportionately costly" — expressible with the cross-product feature
# in a linear VFA; their MILP encoding uses (E_price/10)*(pen_x/3) which is
# linear in the epigraph variable pen_x under the CE substitution.
#
# The Ridge intercept is fitted separately (fit_intercept=True) and stored as
# the 13th component of eta[t]:  VFA = phi(x)^T eta[:12] + eta[12].
# ==============================================================================
N_FEAT = 12


def _phi(state: dict) -> np.ndarray:
    """12-feature state vector. eta[12] holds the Ridge intercept."""
    T1     = float(state['T1'])
    T2     = float(state['T2'])
    pen1   = max(0.0, 22.0 - T1) / 3.0
    pen2   = max(0.0, 22.0 - T2) / 3.0
    p_norm = float(state['price']) / 10.0
    return np.array([
        (T1                                                           - 22.0) /  8.0,
        (T2                                                           - 22.0) /  8.0,
        (float(state.get('H', 0.0))                                   - 40.0) / 40.0,
         p_norm,
         float(state.get('price_previous', state['price']))                   / 10.0,
        (float(state.get('occ1', 0.0))                                - 30.0) / 30.0,
        (float(state.get('occ2', 0.0))                                - 20.0) / 20.0,
         float(state.get('c', 0))                                              /  3.0,
         pen1,
         pen2,
         p_norm * pen1,
         p_norm * pen2,
    ])


# ==============================================================================
# 3. LINEAR VFA WEIGHTS  (produced by ADP_policy_14.ipynb)
#
# Each eta[t] is a 13-vector: [w_0 .. w_11, intercept].
# ==============================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS_PATH = os.path.join(_HERE, 'output', 'adp_weights.json')

try:
    with open(_WEIGHTS_PATH) as _f:
        _w = json.load(_f)
    ETA = {int(t): _w['eta'][t] for t in _w['eta']}
    _n_fixed = 0
    for _t, _v in ETA.items():
        for _i, _val in enumerate(_v):
            if not np.isfinite(_val):
                _v[_i] = 0.0
                _n_fixed += 1
    if _n_fixed:
        print(f"[ADP] WARNING: Replaced {_n_fixed} NaN/Inf weight values with 0.0")
    print(f"[ADP] Loaded {len(ETA)} timesteps × {len(next(iter(ETA.values())))} values "
          f"({N_FEAT} weights + 1 intercept)")
    for _t in sorted(ETA)[:3]:
        print(f"[ADP]   t={_t}: {np.round(ETA[_t], 4)}")
except FileNotFoundError:
    print(f"[ADP] WARNING: {_WEIGHTS_PATH} not found — using zero weights")
    ETA = {t: [0.0] * (N_FEAT + 1) for t in range(10)}  # 13 zeros per timestep

# ==============================================================================
# 4. CERTAINTY-EQUIVALENCE — EXPECTED NEXT-PERIOD EXOGENOUS VALUES
#
# Because the VFA is strictly linear in x and the dynamics are linear in
# (occ1, occ2, price), Jensen's inequality holds with equality:
#     E_w[V(f(x, u, w))] = V(f(x, u, E[w])).
# K=50 Monte Carlo samples from the known Uniform distributions approximate
# E[w], converting the stochastic look-ahead into a deterministic scalar.
# ==============================================================================
K = 50
_OCC1_LO,  _OCC1_HI  = 25, 35
_OCC2_LO,  _OCC2_HI  = 15, 25
_PRICE_LO, _PRICE_HI =  2,  8

_rng = np.random.default_rng(seed=None)


def _expected_exogenous():
    """Return (E[occ1], E[occ2], E[price]) from K Monte Carlo draws."""
    return (
        float(np.mean(_rng.uniform(_OCC1_LO,  _OCC1_HI,  K))),
        float(np.mean(_rng.uniform(_OCC2_LO,  _OCC2_HI,  K))),
        float(np.mean(_rng.uniform(_PRICE_LO, _PRICE_HI, K))),
    )


# ==============================================================================
# 5. ONE-STEP LINEAR MILP
# ==============================================================================
def solve_adp_step(state: dict, eta: list, params: dict) -> dict:
    """
    Solve the one-step MILP under the certainty-equivalence principle.

    Objective (minimised over p1, p2 in [0, P_max] and v in {0, 1}):

        price * (p1 + p2 + P_vent * v)
        + phi(E[x_{t+1}])^T * eta[:12] + eta[12]
        + Big-M overrule penalties

    Features [0,1,2,7,8,9] of phi(E[x_{t+1}]) are Pyomo affine expressions in
    the decision variables; features [3,4,5,6] are scalar constants under the
    certainty-equivalence substitution, evaluated in pure Python before the
    MILP is assembled.  Hockey-stick features [8,9] use auxiliary NonNegativeReal
    variables with linear lower-bound constraints to preserve MILP structure.

    Dynamics use the realised current occupancy (known from the state dict).
    Expected next-period values (K=50 MC samples) are used only for the
    exogenous features of phi(E[x_{t+1}]).
    """
    p     = params
    P_max = p['P_max']
    t     = int(state.get('current_time', 0))
    T_out = float(p['T_out'][min(t, 9)])
    T1    = float(state['T1'])
    T2    = float(state['T2'])
    H     = float(state.get('H', 0.0))
    c     = int(state.get('c', 0))
    occ1  = float(state.get('occ1', 0.0))
    occ2  = float(state.get('occ2', 0.0))
    price = float(state['price'])

    # Expected next-period exogenous values (certainty equivalence)
    exp_occ1_next, exp_occ2_next, exp_price_next = _expected_exogenous()

    m = pyo.ConcreteModel()

    # Decision variables
    m.p1 = pyo.Var(bounds=(0, P_max))
    m.p2 = pyo.Var(bounds=(0, P_max))
    m.v  = pyo.Var(domain=pyo.Binary)

    # Post-decision state variables (auxiliary, determined by dynamics)
    m.T1x = pyo.Var()
    m.T2x = pyo.Var()
    m.Hx  = pyo.Var()

    # Thermal and humidity dynamics (realised current occupancy)
    m.dT1 = pyo.Constraint(expr=
        m.T1x == T1 + p['zeta_exch']*(T2 - T1) + p['zeta_loss']*(T_out - T1)
               + p['zeta_conv']*m.p1 - p['zeta_cool']*m.v + p['zeta_occ']*occ1)
    m.dT2 = pyo.Constraint(expr=
        m.T2x == T2 + p['zeta_exch']*(T1 - T2) + p['zeta_loss']*(T_out - T2)
               + p['zeta_conv']*m.p2 - p['zeta_cool']*m.v + p['zeta_occ']*occ2)
    m.dH  = pyo.Constraint(expr=
        m.Hx  == H + p['eta_occ']*(occ1 + occ2) - p['eta_vent']*m.v)

    # Ventilation inertia counter: normalised feature c_next/3, linear in v.
    # c=0 (free):   c_next = 2*v       → c_next/3 = (2/3)*v
    # c=1 (forced): c_next = 0         → c_next/3 = 0
    # c=2 (forced): c_next = 1         → c_next/3 = 1/3
    if   c == 0: c_next_norm = (2.0 / 3.0) * m.v
    elif c == 1: c_next_norm = 0.0
    else:        c_next_norm = 1.0 / 3.0

    # Asymmetric penalty auxiliaries: epigraph encoding of max(0, 22.0 - T_r_next) / 3.
    # Threshold at T_ok=22°C gives the VFA an early-warning gradient as temperatures
    # drop below the comfort target, well before the hard constraint T_low=18°C.
    # Upper bound 30: pen_x=30 ↔ T_r_next=22-30=-8°C — physically impossible.
    # Physical worst case at T1x≈2°C: pen_x≈(22-2)/3≈6.7, well below the cap.
    m.pen1_x = pyo.Var(bounds=(0.0, 30.0))
    m.pen2_x = pyo.Var(bounds=(0.0, 30.0))
    m.c_pen1 = pyo.Constraint(expr=m.pen1_x >= 22.0 - m.T1x)
    m.c_pen2 = pyo.Constraint(expr=m.pen2_x >= 22.0 - m.T2x)

    # VFA of expected next state: phi(E[x_{t+1}])^T * eta[:12] + eta[12]
    # phi features [0,1,2,7,8,9,10,11]: Pyomo expressions (linear in p1, p2, v).
    # phi features [3,4,5,6]: constants evaluated from CE substitution.
    # Cross-product terms [10,11]: (E_price/10)*pen_x/3 — linear in pen_x (CE makes price constant).
    e_p_norm = exp_price_next / 10.0
    vfa_next = (
        eta[12]                                                     +  # Ridge intercept
        eta[0]  * ((m.T1x         - 22.0) /  8.0)                  +
        eta[1]  * ((m.T2x         - 22.0) /  8.0)                  +
        eta[2]  * ((m.Hx          - 40.0) / 40.0)                  +
        eta[3]  * ( exp_price_next          / 10.0)                 +
        eta[4]  * ( price                   / 10.0)                 +
        eta[5]  * ((exp_occ1_next - 30.0) / 30.0)                  +
        eta[6]  * ((exp_occ2_next - 20.0) / 20.0)                  +
        eta[7]  *  c_next_norm                                      +
        eta[8]  * (m.pen1_x / 3.0)                                 +
        eta[9]  * (m.pen2_x / 3.0)                                 +
        eta[10] * e_p_norm * (m.pen1_x / 3.0)                      +
        eta[11] * e_p_norm * (m.pen2_x / 3.0)
    )

    # Soft Big-M penalties for mandatory overrule actions
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

    result = solver.solve(m)

    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(
            f"Solver failed at t={t}: {result.solver.termination_condition}")

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
    vfa   = float(np.dot(_phi(state), eta[:N_FEAT]) + eta[N_FEAT])  # eta[12] = intercept
    print(f"  t={t} | p1={p1:.2f} p2={p2:.2f} v={v}"
          f" | imm={imm:.3f} vfa_curr={vfa:.2f} | price={price:.2f}"
          f" | T1={state['T1']:.1f} T2={state['T2']:.1f}")

    return decisions