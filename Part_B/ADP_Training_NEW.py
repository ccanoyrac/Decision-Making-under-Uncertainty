"""
ADP_Training_NEW.py
===================
Approximate Policy Iteration (API) with time-indexed Ridge VFA.
Adapted from the other group's version to Group 14's state format,
dynamics, and data sources.

Mathematical algorithm preserved (unchanged):
  - Approximate Policy Iteration: forward pass + backward sweep
  - Time-indexed VFA:  V_t(s) ≈ w_t^T · normalize(s)   for t = 0 … 9
  - Ridge regression (alpha=1) for weight update
  - Policy mixing:  w <- (1-beta)*w_old + beta*w_new

State-key mapping from original -> Group 14:
  price_t           -> price
  price_previous    -> price_previous   (unchanged)
  Occ1 / Occ2      -> occ1 / occ2
  vent_counter      -> c
  low_override_r1/2 -> y_low_1/2

Vent-counter semantics (our 3-step minimum up-time):
  v=1, c=0  ->  c_n = 2   (just turned on; 2 forced steps ahead)
  v=1, c>0  ->  c_n = max(0, c-1)
  v=0       ->  c_n = 0

Output:  vfa_weights[t] for t=0..9  (printed + saved to api_vfa_weights.json)
"""

import json
import numpy as np
from pyomo.environ import *
from sklearn.linear_model import Ridge
import matplotlib
matplotlib.use('Agg')

np.random.seed(42)   # fix 4: reproducibility

from v2_SystemCharacteristics   import get_fixed_data
from PriceProcessRestaurant     import price_model
from OccupancyProcessRestaurant import next_occupancy_levels

# ── HYPERPARAMETERS ────────────────────────────────────────────────────────────
N_SAMPLES            = 100   # trajectories per forward pass (more data → lower regression variance)
# K_SCENARIOS is the number of (price, occ) draws used in the MILP look-ahead.
# NOTE: in the current formulation these draws produce a CONSTANT offset in the
# objective (avg_stochastic does not contain any Pyomo variable), so they have
# zero effect on the optimal (p1, p2, v).  Setting to 1 avoids wasted calls.
K_SCENARIOS          = 1     # 1 is sufficient — scenarios don't affect MILP decisions
K_SCENARIOS_BACKWARD = 50    # MC scenarios in backward-pass target estimate
                              # (these DO affect the regression targets — keep large)
ITERATIONS_I         = 30    # outer API iterations (more needed to escape greedy-policy basin)
T_HOURS              = 10
SWEEPS_J             = 6     # inner policy-evaluation sweeps per iteration
BETA                 = 0.25  # policy-mixing coefficient

# ── SYSTEM PARAMETERS ─────────────────────────────────────────────────────────
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

# ── FEATURE COLUMNS (Group 14 state keys) ─────────────────────────────────────
# fix 2: y_low_1/y_low_2 removed — they cannot be included in the MILP objective
# without Big-M binary variables, so training them only wastes Ridge capacity.
# The temperature features T1/T2 already capture the threshold effect linearly.
feature_cols = [
    'T1', 'T2', 'H',
    'price', 'price_previous',
    'occ1', 'occ2',
    'c',
]

# ── VFA WEIGHT INITIALISATION ─────────────────────────────────────────────────
vfa_weights = {}
for t in range(T_HOURS):
    vfa_weights[t] = {feat: 0.0 for feat in feature_cols}
    vfa_weights[t]['intercept'] = 0.0


# ============================================================================
# NORMALISATION
# ============================================================================
def _norm(state):
    """Normalise state features to roughly [-1, 1] for stable Ridge regression."""
    return {
        'T1'            : (float(state['T1'])             - 22.0) / 8.0,
        'T2'            : (float(state['T2'])             - 22.0) / 8.0,
        'H'             : (float(state['H'])              - 40.0) / 40.0,
        'price'         :  float(state['price'])                  / 10.0,
        'price_previous':  float(state['price_previous'])         / 10.0,
        'occ1'          : (float(state['occ1'])           - 20.0) / 30.0,
        'occ2'          : (float(state['occ2'])           - 10.0) / 20.0,
        'c'             :  float(state['c'])                      /  3.0,
    }


def _vfa(weights, norm_state):
    """Evaluate linear VFA: intercept + w^T * norm(state)."""
    return weights['intercept'] + sum(
        weights[f] * norm_state[f] for f in feature_cols
    )


# ============================================================================
# PHYSICAL DYNAMICS  (replaces EnvFunctions.apply_dynamics)
# ============================================================================
def _apply_phys(state, action):
    """
    Apply action, enforce overrule controllers, advance physical state.
    Price / occupancy are NOT updated here -- caller sets them after this call.
    Returns (next_state, step_cost).
    """
    p1 = float(action['HeatPowerRoom1'])
    p2 = float(action['HeatPowerRoom2'])
    v  = int(float(action['VentilationON']) > 0.5)
    c  = int(state['c'])
    H  = float(state['H'])

    # Overrule controllers (Group 14 semantics -- same as Task6_Environment)
    if c > 0:                    v  = 1
    if H >= _d['H_high']:        v  = 1
    if state['y_low_1']:         p1 = _d['P_max']
    if state['y_low_2']:         p2 = _d['P_max']
    if state.get('y_high_1', 0): p1 = 0.0
    if state.get('y_high_2', 0): p2 = 0.0

    t     = int(state['current_time'])
    T_out = float(_d['T_out'][min(t, 9)])
    T1, T2 = float(state['T1']), float(state['T2'])
    occ1, occ2 = float(state['occ1']), float(state['occ2'])

    cost = float(state['price']) * (p1 + p2 + _d['P_vent'] * v)

    T1_n = T1 + _d['zeta_exch']*(T2-T1) + _d['zeta_loss']*(T_out-T1) + _d['zeta_conv']*p1 - _d['zeta_cool']*v + _d['zeta_occ']*occ1
    T2_n = T2 + _d['zeta_exch']*(T1-T2) + _d['zeta_loss']*(T_out-T2) + _d['zeta_conv']*p2 - _d['zeta_cool']*v + _d['zeta_occ']*occ2
    H_n  = max(0.0, H + _d['eta_occ']*(occ1+occ2) - _d['eta_vent']*v)

    # 3-step minimum up-time counter (Group 14: 2->1->0)
    c_n = (2 if c == 0 else max(0, c - 1)) if v == 1 else 0

    # Low-temperature override with hysteresis
    y1 = 1 if T1_n < _d['T_low'] else (0 if T1_n >= _d['T_OK'] else int(state['y_low_1']))
    y2 = 1 if T2_n < _d['T_low'] else (0 if T2_n >= _d['T_OK'] else int(state['y_low_2']))

    next_state = {
        'T1': T1_n, 'T2': T2_n, 'H': H_n,
        'c': c_n, 'y_low_1': y1, 'y_low_2': y2,
        'y_high_1': int(T1_n > _d['T_high']),
        'y_high_2': int(T2_n > _d['T_high']),
        'price'         : float(state['price']),
        'price_previous': float(state['price_previous']),
        'occ1': occ1, 'occ2': occ2,
        'current_time': t + 1,
    }
    return next_state, cost


# ============================================================================
# 1. MILP -- FORWARD PASS
#    Solves: min  r(s,u) + (1/K) sum_k V_{t+1}(s_k^+; w_{t+1})
# ============================================================================
def solve_bellman_milp(state, next_weights):
    """
    Here-and-now MILP with K_SCENARIOS Monte-Carlo look-ahead via the VFA.
    Returns {'HeatPowerRoom1', 'HeatPowerRoom2', 'VentilationON'}.
    """
    t      = int(state['current_time'])
    T_out  = float(_d['T_out'][min(t, 9)])
    T1, T2 = float(state['T1']), float(state['T2'])
    H      = float(state['H'])
    occ1, occ2 = float(state['occ1']), float(state['occ2'])
    price  = float(state['price'])
    c      = int(state['c'])

    m = ConcreteModel()
    m.p1 = Var(bounds=(0.0, _d['P_max']))
    m.p2 = Var(bounds=(0.0, _d['P_max']))
    m.v  = Var(domain=Binary)

    # Hard overrule constraints (Group 14 semantics)
    if c > 0:                    m.v.fix(1)
    if H >= _d['H_high']:        m.v.fix(1)
    if state['y_low_1']:         m.p1.fix(_d['P_max'])
    if state['y_low_2']:         m.p2.fix(_d['P_max'])
    if state.get('y_high_1', 0): m.p1.fix(0.0)
    if state.get('y_high_2', 0): m.p2.fix(0.0)

    # Next physical state (deterministic given action -- no uncertainty in dynamics)
    m.T1x = Var(bounds=(-20.0, 60.0))
    m.T2x = Var(bounds=(-20.0, 60.0))
    m.Hx  = Var(bounds=(0.0, 200.0))
    m.cT1 = Constraint(expr=m.T1x == T1 + _d['zeta_exch']*(T2-T1) + _d['zeta_loss']*(T_out-T1) + _d['zeta_conv']*m.p1 - _d['zeta_cool']*m.v + _d['zeta_occ']*occ1)
    m.cT2 = Constraint(expr=m.T2x == T2 + _d['zeta_exch']*(T1-T2) + _d['zeta_loss']*(T_out-T2) + _d['zeta_conv']*m.p2 - _d['zeta_cool']*m.v + _d['zeta_occ']*occ2)
    m.cH  = Constraint(expr=m.Hx  == H  + _d['eta_occ']*(occ1+occ2) - _d['eta_vent']*m.v)

    # Next vent counter (linear in v for our 3-step semantics)
    # c=0 (free): c_n = 2*v;   c>0 (forced, v=1 already fixed): c_n = max(0,c-1)
    vc_next = (2.0 * m.v) if c == 0 else float(max(0, c - 1))

    immediate_cost = price * (m.p1 + m.p2 + _d['P_vent'] * m.v)

    expected_future = 0.0
    if next_weights:
        # Pre-generate K_SCENARIOS stochastic (price, occ) futures
        scenarios = []
        for _ in range(K_SCENARIOS):
            sc_p         = price_model(state['price'], state['price_previous'])
            sc_o1, sc_o2 = next_occupancy_levels(state['occ1'], state['occ2'])
            scenarios.append({'price': sc_p, 'occ1': sc_o1, 'occ2': sc_o2})

        # VFA terms that depend on MILP variables (factor out of the scenario sum)
        vfa_phys = (
              next_weights['T1'] * ((m.T1x - 22.0) / 8.0)
            + next_weights['T2'] * ((m.T2x - 22.0) / 8.0)
            + next_weights['H']  * ((m.Hx  - 40.0) / 40.0)
            + next_weights['c']  * (vc_next          / 3.0)
        )
        # price_previous in next state = current price (deterministic constant)
        vfa_prev_price = next_weights['price_previous'] * (price / 10.0)

        # Average over scenario-dependent terms (stochastic -- constants in MILP)
        avg_stochastic = (1.0 / K_SCENARIOS) * sum(
              next_weights['price'] * (sc['price'] / 10.0)
            + next_weights['occ1']  * ((sc['occ1'] - 20.0) / 30.0)
            + next_weights['occ2']  * ((sc['occ2'] - 10.0) / 20.0)
            for sc in scenarios
        )
        # y_low flags in next state depend on T1x/T2x via hysteresis logic;
        # to avoid Big-M binary variables, their contribution is set to 0 here
        # (Ridge regression captures the temperature effect via T1/T2 weights).

        expected_future = (
            next_weights['intercept']
            + vfa_phys
            + vfa_prev_price
            + avg_stochastic
        )

    m.obj = Objective(expr=immediate_cost + expected_future, sense=minimize)
    SolverFactory('gurobi').solve(m, tee=False)

    try:
        p1_v = float(value(m.p1))
        p2_v = float(value(m.p2))
        v_v  = int(round(float(value(m.v))))
    except Exception:
        p1_v = _d['P_max'] if state['y_low_1'] else 0.0
        p2_v = _d['P_max'] if state['y_low_2'] else 0.0
        v_v  = 1 if (c > 0 or H >= _d['H_high']) else 0

    return {'HeatPowerRoom1': p1_v, 'HeatPowerRoom2': p2_v, 'VentilationON': v_v}


# ============================================================================
# 2. BACKWARD-PASS TARGET EVALUATION
#    Computes V*(x_t) = r(x_t, u_t) + E[ V^(x_{t+1}; eta^j) ]
# ============================================================================
def evaluate_fixed_action(state, action, next_weights):
    """
    Compute target value for a visited (state, action) pair using
    K_SCENARIOS_BACKWARD MC samples of next price/occ.
    """
    p1 = float(action['HeatPowerRoom1'])
    p2 = float(action['HeatPowerRoom2'])
    v  = int(action['VentilationON'])
    c  = int(state['c'])
    H  = float(state['H'])

    # Enforce overrule controllers
    if c > 0:                    v  = 1
    if H >= _d['H_high']:        v  = 1
    if state['y_low_1']:         p1 = _d['P_max']
    if state['y_low_2']:         p2 = _d['P_max']
    if state.get('y_high_1', 0): p1 = 0.0
    if state.get('y_high_2', 0): p2 = 0.0

    imm_cost = float(state['price']) * (p1 + p2 + _d['P_vent'] * v)
    if next_weights is None:
        return imm_cost

    # Deterministic next physical state
    t     = int(state['current_time'])
    T_out = float(_d['T_out'][min(t, 9)])
    T1, T2 = float(state['T1']), float(state['T2'])
    occ1, occ2 = float(state['occ1']), float(state['occ2'])

    T1_n = T1 + _d['zeta_exch']*(T2-T1) + _d['zeta_loss']*(T_out-T1) + _d['zeta_conv']*p1 - _d['zeta_cool']*v + _d['zeta_occ']*occ1
    T2_n = T2 + _d['zeta_exch']*(T1-T2) + _d['zeta_loss']*(T_out-T2) + _d['zeta_conv']*p2 - _d['zeta_cool']*v + _d['zeta_occ']*occ2
    H_n  = max(0.0, H + _d['eta_occ']*(occ1+occ2) - _d['eta_vent']*v)
    c_n  = (2 if c == 0 else max(0, c - 1)) if v == 1 else 0
    y1   = 1 if T1_n < _d['T_low'] else (0 if T1_n >= _d['T_OK'] else int(state['y_low_1']))
    y2   = 1 if T2_n < _d['T_low'] else (0 if T2_n >= _d['T_OK'] else int(state['y_low_2']))

    # MC expectation over stochastic next price/occ
    expected_vfa = 0.0
    for _ in range(K_SCENARIOS_BACKWARD):
        sc_p         = price_model(float(state['price']), float(state['price_previous']))
        sc_o1, sc_o2 = next_occupancy_levels(occ1, occ2)
        next_sim = {
            'T1': T1_n, 'T2': T2_n, 'H': H_n,
            'c': c_n, 'y_low_1': y1, 'y_low_2': y2,
            'price': sc_p, 'price_previous': state['price'],
            'occ1': sc_o1, 'occ2': sc_o2,
        }
        expected_vfa += (1.0 / K_SCENARIOS_BACKWARD) * _vfa(next_weights, _norm(next_sim))

    return imm_cost + expected_vfa


# ============================================================================
# MAIN TRAINING LOOP -- APPROXIMATE POLICY ITERATION
# ============================================================================
for i in range(ITERATIONS_I):
    print(f"\n=== OUTER LOOP i={i+1}/{ITERATIONS_I} (Policy Improvement) ===")

    visited = {t: [] for t in range(T_HOURS)}

    # -- FORWARD PASS: generate N_SAMPLES trajectories ------------------------
    current_states = []
    for n in range(N_SAMPLES):
        # fix 3: sample from realistic operational ranges that match deployment.
        # Deployment always starts at T=21, H=40; keep training close but diverse.
        # T range [19,24]: stays inside comfort zone [18,26] with some margin.
        # H range [30,60]: avoids the unrealistic H<20 region seen in old version.
        s = {
            'T1'            : np.random.uniform(19.0, 24.0),
            'T2'            : np.random.uniform(19.0, 24.0),
            'H'             : np.random.uniform(30.0, 60.0),
            'occ1'          : np.random.uniform(25.0, 35.0),
            'occ2'          : np.random.uniform(15.0, 25.0),
            'price'         : np.random.uniform(0.0,  12.0),
            'price_previous': np.random.uniform(0.0,  12.0),
            'c'             : 0,
            'y_low_1'       : 0,
            'y_low_2'       : 0,
            'y_high_1'      : 0,
            'y_high_2'      : 0,
            'current_time'  : 0,
        }
        current_states.append(s)

    for t in range(T_HOURS):
        next_w = vfa_weights[t + 1] if t < T_HOURS - 1 else None
        for n in range(N_SAMPLES):
            state_n = current_states[n]
            state_n['current_time'] = t

            action = solve_bellman_milp(state_n, next_w)
            visited[t].append((state_n.copy(), action))

            next_n, _ = _apply_phys(state_n, action)

            if t + 1 < T_HOURS:
                new_o1, new_o2 = next_occupancy_levels(state_n['occ1'], state_n['occ2'])
                new_p          = price_model(state_n['price'], state_n['price_previous'])
                next_n['occ1']           = new_o1
                next_n['occ2']           = new_o2
                next_n['price_previous'] = state_n['price']
                next_n['price']          = new_p

            current_states[n] = next_n

    # -- BACKWARD PASS + INNER SWEEPS -----------------------------------------
    inner_weights = {t: dict(vfa_weights[t]) for t in range(T_HOURS)}

    for j in range(SWEEPS_J):
        print(f"  Inner sweep j={j+1}/{SWEEPS_J}")
        for t in reversed(range(T_HOURS)):
            next_w = inner_weights[t + 1] if t < T_HOURS - 1 else None
            X, Y   = [], []

            for state_n, action_n in visited[t]:
                Y.append(evaluate_fixed_action(state_n, action_n, next_w))
                nf = _norm(state_n)
                X.append([nf[f] for f in feature_cols])

            if X:
                reg = Ridge(alpha=1.0, fit_intercept=True)
                reg.fit(X, Y)
                for idx, feat in enumerate(feature_cols):
                    inner_weights[t][feat] = float(reg.coef_[idx])
                inner_weights[t]['intercept'] = float(reg.intercept_)

    # -- POLICY MIXING ---------------------------------------------------------
    for t in range(T_HOURS):
        for k in feature_cols + ['intercept']:
            vfa_weights[t][k] = (1 - BETA) * vfa_weights[t][k] + BETA * inner_weights[t][k]

    if (i + 1) % 5 == 0 or i == 0:
        print(f"\n  Snapshot at iteration {i+1}:")
        for t in range(T_HOURS):
            print(f"    t={t}  intercept={vfa_weights[t]['intercept']:+.3f}"
                  f"  T1={vfa_weights[t]['T1']:+.3f}  T2={vfa_weights[t]['T2']:+.3f}"
                  f"  price={vfa_weights[t]['price']:+.3f}"
                  f"  c={vfa_weights[t]['c']:+.3f}")


# ── FINAL OUTPUT ───────────────────────────────────────────────────────────────
print("\n=== FINAL VFA_WEIGHTS ===")
print("VFA_WEIGHTS = {")
for t in range(T_HOURS):
    clean = {k: round(float(v), 4) for k, v in vfa_weights[t].items()}
    print(f"    {t}: {clean},")
print("}")

out = {str(t): {k: float(v) for k, v in vfa_weights[t].items()}
       for t in range(T_HOURS)}
with open('api_vfa_weights.json', 'w') as f:
    json.dump(out, f, indent=2)
print("\nSaved to api_vfa_weights.json")
