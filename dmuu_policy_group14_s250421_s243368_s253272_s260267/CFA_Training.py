"""
CFA_Training.py
===============
Approximate Policy Iteration (API) with time-indexed Ridge VFA.
Adapted to Group 14's environment (Task6_Environment.apply_dynamics).

Mathematical algorithm preserved:
  - Approximate Policy Iteration: forward pass + backward sweep
  - Time-indexed VFA:  V_t(s) ≈ w_t^T · normalize(s)   for t = 0 … 9
  - Ridge regression (alpha=1) for weight update
  - Policy mixing:  w <- (1-beta)*w_old + beta*w_new

Dynamics:  Task6_Environment.apply_dynamics — single source of truth.

Output:  vfa_weights[t] for t=0..9  (printed + saved to Data/CFA_weights.csv)
"""

import csv
import os
import numpy as np
from pyomo.environ import *
from sklearn.linear_model import Ridge

np.random.seed(42)

from Data.v2_SystemCharacteristics   import get_fixed_data
from Data.PriceProcessRestaurant     import price_model
from Data.OccupancyProcessRestaurant import next_occupancy_levels
from Task6_Environment import apply_dynamics, run_policy, dummy_action

# ── HYPERPARAMETERS ────────────────────────────────────────────────────────────
N_SAMPLES            = 100
K_SCENARIOS          = 1
K_SCENARIOS_BACKWARD = 50
ITERATIONS_I         = 30
T_HOURS              = 10
SWEEPS_J             = 6
BETA                 = 0.25

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

# ── FEATURE COLUMNS ────────────────────────────────────────────────────────────
feature_cols = [
    'T1', 'T2', 'H',
    'price_t', 'price_previous',
    'Occ1', 'Occ2',
    'vent_counter',
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
    return {
        'T1'            : (float(state['T1'])             - 22.0) / 8.0,
        'T2'            : (float(state['T2'])             - 22.0) / 8.0,
        'H'             : (float(state['H'])              - 40.0) / 40.0,
        'price_t'       :  float(state['price_t'])                 / 10.0,
        'price_previous':  float(state['price_previous'])          / 10.0,
        'Occ1'          : (float(state['Occ1'])           - 20.0) / 30.0,
        'Occ2'          : (float(state['Occ2'])           - 10.0) / 20.0,
        'vent_counter'  :  float(state['vent_counter'])            /  3.0,
    }


def _vfa(weights, norm_state):
    return weights['intercept'] + sum(
        weights[f] * norm_state[f] for f in feature_cols
    )


# ============================================================================
# 1. MILP -- FORWARD PASS
# ============================================================================
def solve_bellman_milp(state, next_weights):
    t      = int(state['current_time'])
    T_out  = float(_d['T_out'][min(t, 9)])
    T1, T2 = float(state['T1']), float(state['T2'])
    H      = float(state['H'])
    occ1, occ2 = float(state['Occ1']), float(state['Occ2'])
    price  = float(state['price_t'])
    c      = int(state['vent_counter'])

    m = ConcreteModel()
    m.p1 = Var(bounds=(0.0, _d['P_max']))
    m.p2 = Var(bounds=(0.0, _d['P_max']))
    m.v  = Var(domain=Binary)

    if c > 0:                          m.v.fix(1)
    if H >= _d['H_high']:              m.v.fix(1)
    if state.get('low_override_r1'):   m.p1.fix(_d['P_max'])
    if state.get('low_override_r2'):   m.p2.fix(_d['P_max'])
    if state.get('y_high_1'):          m.p1.fix(0.0)
    if state.get('y_high_2'):          m.p2.fix(0.0)

    m.T1x = Var(bounds=(-20.0,  60.0))
    m.T2x = Var(bounds=(-20.0,  60.0))
    m.Hx  = Var(bounds=(-200.0, 200.0))
    m.cT1 = Constraint(expr=m.T1x == T1 + _d['zeta_exch']*(T2-T1) + _d['zeta_loss']*(T_out-T1) + _d['zeta_conv']*m.p1 - _d['zeta_cool']*m.v + _d['zeta_occ']*occ1)
    m.cT2 = Constraint(expr=m.T2x == T2 + _d['zeta_exch']*(T1-T2) + _d['zeta_loss']*(T_out-T2) + _d['zeta_conv']*m.p2 - _d['zeta_cool']*m.v + _d['zeta_occ']*occ2)
    m.cH  = Constraint(expr=m.Hx  == H  + _d['eta_occ']*(occ1+occ2) - _d['eta_vent']*m.v)

    vc_next = (2.0 * m.v) if c == 0 else float(max(0, c - 1))

    immediate_cost = price * (m.p1 + m.p2 + _d['P_vent'] * m.v)

    expected_future = 0.0
    if next_weights:
        scenarios = []
        for _ in range(K_SCENARIOS):
            sc_p         = price_model(state['price_t'], state['price_previous'])
            sc_o1, sc_o2 = next_occupancy_levels(state['Occ1'], state['Occ2'])
            scenarios.append({'price': sc_p, 'occ1': sc_o1, 'occ2': sc_o2})

        vfa_phys = (
              next_weights['T1']           * ((m.T1x - 22.0) / 8.0)
            + next_weights['T2']           * ((m.T2x - 22.0) / 8.0)
            + next_weights['H']            * ((m.Hx  - 40.0) / 40.0)
            + next_weights['vent_counter'] * (vc_next / 3.0)
        )
        vfa_prev_price = next_weights['price_previous'] * (price / 10.0)

        avg_stochastic = (1.0 / K_SCENARIOS) * sum(
              next_weights['price_t'] * (sc['price'] / 10.0)
            + next_weights['Occ1']    * ((sc['occ1'] - 20.0) / 30.0)
            + next_weights['Occ2']    * ((sc['occ2'] - 10.0) / 20.0)
            for sc in scenarios
        )

        expected_future = (
            next_weights['intercept']
            + vfa_phys
            + vfa_prev_price
            + avg_stochastic
        )

    m.obj = Objective(expr=immediate_cost + expected_future, sense=minimize)

    slv = SolverFactory('gurobi')
    slv.options['OutputFlag'] = 0
    res = slv.solve(m)

    _fallback = {
        'HeatPowerRoom1': _d['P_max'] if state.get('low_override_r1') else 0.0,
        'HeatPowerRoom2': _d['P_max'] if state.get('low_override_r2') else 0.0,
        'VentilationON' : 1 if (c > 0 or H >= _d['H_high']) else 0,
    }

    ok = (TerminationCondition.optimal, TerminationCondition.feasible)
    if res.solver.termination_condition not in ok:
        return _fallback

    try:
        return {
            'HeatPowerRoom1': float(value(m.p1)),
            'HeatPowerRoom2': float(value(m.p2)),
            'VentilationON' : int(round(float(value(m.v)))),
        }
    except Exception:
        return _fallback


# ============================================================================
# 2. BACKWARD-PASS TARGET EVALUATION
# ============================================================================
def evaluate_fixed_action(state, action, next_weights):
    next_state, imm_cost = apply_dynamics(state, action)

    if next_weights is None:
        return imm_cost

    expected_vfa = 0.0
    for _ in range(K_SCENARIOS_BACKWARD):
        sc_p         = price_model(float(state['price_t']), float(state['price_previous']))
        sc_o1, sc_o2 = next_occupancy_levels(float(state['Occ1']), float(state['Occ2']))
        next_sim = dict(next_state)
        next_sim['price_t'] = sc_p
        next_sim['Occ1']    = sc_o1
        next_sim['Occ2']    = sc_o2
        expected_vfa += (1.0 / K_SCENARIOS_BACKWARD) * _vfa(next_weights, _norm(next_sim))

    return imm_cost + expected_vfa


# ============================================================================
# MAIN TRAINING LOOP -- APPROXIMATE POLICY ITERATION
# ============================================================================
for i in range(ITERATIONS_I):
    print(f"\n=== OUTER LOOP i={i+1}/{ITERATIONS_I} (Policy Improvement) ===")

    visited = {t: [] for t in range(T_HOURS)}

    current_states = []
    for n in range(N_SAMPLES):
        s = {
            'T1'            : np.random.uniform(19.0, 24.0),
            'T2'            : np.random.uniform(19.0, 24.0),
            'H'             : np.random.uniform(30.0, 60.0),
            'Occ1'          : np.random.uniform(25.0, 35.0),
            'Occ2'          : np.random.uniform(15.0, 25.0),
            'price_t'       : np.random.uniform(0.0,  12.0),
            'price_previous': np.random.uniform(0.0,  12.0),
            'vent_counter'  : 0,
            'low_override_r1': 0,
            'low_override_r2': 0,
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

            next_n, _ = apply_dynamics(state_n, action)

            if t + 1 < T_HOURS:
                new_o1, new_o2           = next_occupancy_levels(state_n['Occ1'], state_n['Occ2'])
                new_p                    = price_model(state_n['price_t'], state_n['price_previous'])
                next_n['Occ1']           = new_o1
                next_n['Occ2']           = new_o2
                next_n['price_previous'] = state_n['price_t']
                next_n['price_t']        = new_p

            current_states[n] = next_n

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

    for t in range(T_HOURS):
        for k in feature_cols + ['intercept']:
            vfa_weights[t][k] = (1 - BETA) * vfa_weights[t][k] + BETA * inner_weights[t][k]

    if (i + 1) % 5 == 0 or i == 0:
        print(f"\n  Snapshot at iteration {i+1}:")
        for t in range(T_HOURS):
            print(f"    t={t}  intercept={vfa_weights[t]['intercept']:+.3f}"
                  f"  T1={vfa_weights[t]['T1']:+.3f}  T2={vfa_weights[t]['T2']:+.3f}"
                  f"  price_t={vfa_weights[t]['price_t']:+.3f}"
                  f"  vent_counter={vfa_weights[t]['vent_counter']:+.3f}")


# ── SAVE WEIGHTS TO CFA_weights.csv ───────────────────────────────────────────
print("\n=== FINAL VFA_WEIGHTS ===")
for t in range(T_HOURS):
    clean = {k: round(float(v), 4) for k, v in vfa_weights[t].items()}
    print(f"  t={t}: {clean}")

_dir      = os.path.dirname(os.path.abspath(__file__))
_out_path = os.path.join(_dir, 'Data', 'CFA_weights.csv')
_cols     = ['t'] + feature_cols + ['intercept']

with open(_out_path, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=_cols)
    writer.writeheader()
    for t in range(T_HOURS):
        row = {'t': t}
        for feat in feature_cols:
            row[feat] = float(vfa_weights[t][feat])
        row['intercept'] = float(vfa_weights[t]['intercept'])
        writer.writerow(row)

print(f"\nSaved CFA weights to {_out_path}")


# ── EVALUATION ON CSV TEST SET ─────────────────────────────────────────────────
_data_dir = os.path.join(_dir, 'Data')

def _cfa_select_action(state: dict) -> dict:
    t = int(state.get('current_time', 0))
    c = int(state.get('vent_counter', 0))
    H = float(state.get('H', 0.0))

    if t >= 9:
        return {
            'HeatPowerRoom1': _d['P_max'] if state.get('low_override_r1') else 0.0,
            'HeatPowerRoom2': _d['P_max'] if state.get('low_override_r2') else 0.0,
            'VentilationON' : 1 if (c > 0 or H >= _d['H_high']) else 0,
        }

    next_w = vfa_weights.get(t + 1)
    try:
        return solve_bellman_milp(state, next_w)
    except Exception:
        return {
            'HeatPowerRoom1': _d['P_max'] if state.get('low_override_r1') else 0.0,
            'HeatPowerRoom2': _d['P_max'] if state.get('low_override_r2') else 0.0,
            'VentilationON' : 1 if (c > 0 or H >= _d['H_high']) else 0,
        }


price_csv = os.path.join(_data_dir, 'v2_PriceData.csv')
occ1_csv  = os.path.join(_data_dir, 'OccupancyRoom1.csv')
occ2_csv  = os.path.join(_data_dir, 'OccupancyRoom2.csv')

print("\nEvaluating on CSV test set (100 days) ...")
cfa_costs, _ = run_policy(
    _cfa_select_action, price_csv, occ1_csv, occ2_csv,
    policy_name="CFA (trained)", verbose=True,
)
dummy_costs, _ = run_policy(
    dummy_action, price_csv, occ1_csv, occ2_csv,
    policy_name="Dummy", verbose=False,
)

print(f"\nCFA (trained) — mean: {cfa_costs.mean():.3f}  std: {cfa_costs.std():.3f} euro")
print(f"Dummy         — mean: {dummy_costs.mean():.3f}  std: {dummy_costs.std():.3f} euro")
print(f"Improvement   : {(dummy_costs.mean() - cfa_costs.mean()) / dummy_costs.mean() * 100:.1f}% vs dummy")
