import numpy as np
import csv
from pathlib import Path
from pyomo.environ import *
from sklearn.linear_model import Ridge
import sys

for _d in [str(Path("Data").resolve())]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from Task6_Environment import apply_dynamics
from Data.PriceProcessRestaurant import price_model
from Data.OccupancyProcessRestaurant import next_occupancy_levels
from Data.v2_SystemCharacteristics import get_fixed_data

# HYPERPARAMETERS & SETTINGS
N_SAMPLES = 80
K_SCENARIOS = 15
K_SCENARIOS_BACKWARD = 30
ITERATIONS_I = 20
T_HOURS = 10
SWEEPS_J = 6
BETA = 0.25

data = get_fixed_data()
feature_cols = [
    "T1",
    "T2",
    "H",
    "price_t",
    "price_previous",
    "Occ1",
    "Occ2",
    "vent_counter",
    "low_override_r1",
    "low_override_r2",
]

# Initialization
vfa_weights = {}
for t in range(T_HOURS):
    vfa_weights[t] = {feat: 0.0 for feat in feature_cols}
    vfa_weights[t]['intercept'] = 0.0


# ============================================================================
# NORMALISATION
# ============================================================================
def get_normalized_features(state):
    return {
        'T1'            : (state['T1']             - 22.0) / 8.0,
        'T2'            : (state['T2']             - 22.0) / 8.0,
        'H'             : (state['H']              - 40.0) / 40.0,
        'Occ1'          : (state['Occ1']           - 20.0) / 30.0,
        'Occ2'          : (state['Occ2']           - 10.0) / 20.0,
        'price_t'       :  state['price_t']                / 10.0,
        'price_previous':  state['price_previous']         / 10.0,
        'vent_counter'  :  state['vent_counter']            / 3.0,
        'low_override_r1': state['low_override_r1'],
        'low_override_r2': state['low_override_r2'],
    }


# ============================================================================
# 1. MILP FUNCTION
# ============================================================================
def solve_bellman_equation_milp(state, next_t_weights):
    m = ConcreteModel()
    m.p1 = Var(bounds=(0, data['heating_max_power']))
    m.p2 = Var(bounds=(0, data['heating_max_power']))
    m.v  = Var(domain=Binary)

    # Overrule current state
    if state['T1'] > data['temp_max_comfort_threshold']:
        m.p1.fix(0)
    elif state['low_override_r1'] == 1 and state['T1'] < data['temp_OK_threshold']:
        m.p1.fix(data['heating_max_power'])
    elif state['T1'] < data['temp_min_comfort_threshold']:
        m.p1.fix(data['heating_max_power'])

    if state['T2'] > data['temp_max_comfort_threshold']:
        m.p2.fix(0)
    elif state['low_override_r2'] == 1 and state['T2'] < data['temp_OK_threshold']:
        m.p2.fix(data['heating_max_power'])
    elif state['T2'] < data['temp_min_comfort_threshold']:
        m.p2.fix(data['heating_max_power'])

    if state['H'] > data['humidity_threshold'] or state['vent_counter'] in [1, 2]:
        m.v.fix(1)

    scen_data = []
    for _ in range(K_SCENARIOS):
        sc_p = price_model(state['price_t'], state['price_previous'])
        sc_o1, sc_o2 = next_occupancy_levels(state['Occ1'], state['Occ2'])
        scen_data.append({'price_t': sc_p, 'Occ1': sc_o1, 'Occ2': sc_o2})

    m.T1_next = Var()
    m.T2_next = Var()
    m.H_next  = Var()
    m.ov1_next = Var(domain=Binary)
    m.ov2_next = Var(domain=Binary)
    m.vc_next  = Var(domain=NonNegativeReals)
    m.y_low_r1 = Var(domain=Binary); m.y_ok_r1 = Var(domain=Binary)
    m.y_low_r2 = Var(domain=Binary); m.y_ok_r2 = Var(domain=Binary)

    immediate_cost    = state['price_t'] * (m.p1 + m.p2 + m.v * data['ventilation_power'])
    expected_future_cost = 0
    tout = data['outdoor_temperature'][int(state['current_time'])]
    M, eps = 500, 10e-6

    # Physical dynamics
    m.ct1 = Constraint(expr=
        m.T1_next == state['T1'] + data['heat_exchange_coeff']*(state['T2']-state['T1']) +
                     data['thermal_loss_coeff']*(tout-state['T1']) +
                     data['heating_efficiency_coeff']*m.p1 -
                     data['heat_vent_coeff']*m.v +
                     data['heat_occupancy_coeff']*state['Occ1'])

    m.ct2 = Constraint(expr=
        m.T2_next == state['T2'] + data['heat_exchange_coeff']*(state['T1']-state['T2']) +
                     data['thermal_loss_coeff']*(tout-state['T2']) +
                     data['heating_efficiency_coeff']*m.p2 -
                     data['heat_vent_coeff']*m.v +
                     data['heat_occupancy_coeff']*state['Occ2'])

    m.ch = Constraint(expr=
        m.H_next == state['H'] - data['humidity_vent_coeff']*m.v +
                    data['humidity_occupancy_coeff']*(state['Occ1']+state['Occ2']))

    c_curr = int(state['vent_counter'])
    if c_curr == 0:
        m.c_vc = Constraint(expr=m.vc_next == 2 * m.v)
    elif c_curr == 1:
        m.vc_next.fix(0)
    else:
        m.vc_next.fix(1)

    # Override logic next state Room 1
    u_prev_r1 = state['low_override_r1']
    m.c_ylow_r1_a = Constraint(expr=m.T1_next <= data['temp_min_comfort_threshold'] + M*(1 - m.y_low_r1))
    m.c_ylow_r1_b = Constraint(expr=m.T1_next >= data['temp_min_comfort_threshold'] + eps - M*m.y_low_r1)
    m.c_yok_r1_a  = Constraint(expr=m.T1_next >= data['temp_OK_threshold'] - M*(1 - m.y_ok_r1))
    m.c_yok_r1_b  = Constraint(expr=m.T1_next <= data['temp_OK_threshold'] + M*m.y_ok_r1)
    m.c_ov1_a = Constraint(expr=m.ov1_next >= m.y_low_r1)
    m.c_ov1_b = Constraint(expr=m.ov1_next <= u_prev_r1 + m.y_low_r1)
    m.c_ov1_c = Constraint(expr=m.ov1_next >= u_prev_r1 - m.y_ok_r1)
    m.c_ov1_d = Constraint(expr=m.ov1_next <= 1 - m.y_ok_r1)

    # Override logic next state Room 2
    u_prev_r2 = state['low_override_r2']
    m.c_ylow_r2_a = Constraint(expr=m.T2_next <= data['temp_min_comfort_threshold'] + M*(1 - m.y_low_r2))
    m.c_ylow_r2_b = Constraint(expr=m.T2_next >= data['temp_min_comfort_threshold'] + eps - M*m.y_low_r2)
    m.c_yok_r2_a  = Constraint(expr=m.T2_next >= data['temp_OK_threshold'] - M*(1 - m.y_ok_r2))
    m.c_yok_r2_b  = Constraint(expr=m.T2_next <= data['temp_OK_threshold'] + M*m.y_ok_r2)
    m.c_ov2_a = Constraint(expr=m.ov2_next >= m.y_low_r2)
    m.c_ov2_b = Constraint(expr=m.ov2_next <= u_prev_r2 + m.y_low_r2)
    m.c_ov2_c = Constraint(expr=m.ov2_next >= u_prev_r2 - m.y_ok_r2)
    m.c_ov2_d = Constraint(expr=m.ov2_next <= 1 - m.y_ok_r2)

    # Expected future cost via VFA
    if next_t_weights:
        for k in range(K_SCENARIOS):
            expected_future_cost += (1.0 / K_SCENARIOS) * (
                next_t_weights['intercept'] +
                next_t_weights['T1']             * ((m.T1_next - 22.0) / 8.0) +
                next_t_weights['T2']             * ((m.T2_next - 22.0) / 8.0) +
                next_t_weights['H']              * ((m.H_next  - 40.0) / 40.0) +
                next_t_weights['price_t']        * (scen_data[k]['price_t'] / 10.0) +
                next_t_weights['price_previous'] * (state['price_t'] / 10.0) +
                next_t_weights['Occ1']           * ((scen_data[k]['Occ1'] - 20.0) / 30.0) +
                next_t_weights['Occ2']           * ((scen_data[k]['Occ2'] - 10.0) / 20.0) +
                next_t_weights['vent_counter']   * (m.vc_next / 3.0) +
                next_t_weights['low_override_r1']* m.ov1_next +
                next_t_weights['low_override_r2']* m.ov2_next
            )

    m.obj = Objective(expr=immediate_cost + expected_future_cost, sense=minimize)
    SolverFactory('gurobi').solve(m, tee=False)

    return {
        "HeatPowerRoom1": value(m.p1),
        "HeatPowerRoom2": value(m.p2),
        "VentilationON" : int(value(m.v)),
    }


# ============================================================================
# 2. BACKWARD-PASS TARGET EVALUATION
# ============================================================================
def evaluate_fixed_action(state, action, next_weights):
    imm_cost = state['price_t'] * (
        action['HeatPowerRoom1'] + action['HeatPowerRoom2'] +
        action['VentilationON'] * data['ventilation_power']
    )
    if next_weights is None:
        return imm_cost

    expected_vfa = 0.0
    tout = data['outdoor_temperature'][int(state['current_time'])]

    for _ in range(K_SCENARIOS_BACKWARD):
        sc_p = price_model(state['price_t'], state['price_previous'])
        sc_o1, sc_o2 = next_occupancy_levels(state['Occ1'], state['Occ2'])

        T1_n = state['T1'] + data['heat_exchange_coeff'] * (state['T2']-state['T1']) + data['thermal_loss_coeff'] * (tout-state['T1']) + data['heating_efficiency_coeff']*action['HeatPowerRoom1'] - data['heat_vent_coeff']*action['VentilationON'] + data['heat_occupancy_coeff']*state['Occ1']
        T2_n = state['T2'] + data['heat_exchange_coeff'] * (state['T1']-state['T2']) + data['thermal_loss_coeff'] * (tout-state['T2']) + data['heating_efficiency_coeff']*action['HeatPowerRoom2'] - data['heat_vent_coeff']*action['VentilationON'] + data['heat_occupancy_coeff']*state['Occ2']
        H_n  = max(0.0, state['H'] - data['humidity_vent_coeff']*action['VentilationON'] + data['humidity_occupancy_coeff'] * (state['Occ1']+state['Occ2']))
        v_int = int(action['VentilationON'])
        c_int = int(state['vent_counter'])
        vc_n  = (2 if c_int == 0 else max(0, c_int - 1)) if v_int == 1 else 0

        if T1_n < data['temp_min_comfort_threshold']:  ov1_n = 1
        elif T1_n >= data['temp_OK_threshold']:        ov1_n = 0
        else:                                          ov1_n = state['low_override_r1']

        if T2_n < data['temp_min_comfort_threshold']:  ov2_n = 1
        elif T2_n >= data['temp_OK_threshold']:        ov2_n = 0
        else:                                          ov2_n = state['low_override_r2']

        next_state_sim = {
            'T1': T1_n, 'T2': T2_n, 'H': H_n,
            'Occ1': sc_o1, 'Occ2': sc_o2,
            'price_t': sc_p, 'price_previous': state['price_t'],
            'vent_counter': vc_n,
            'low_override_r1': ov1_n, 'low_override_r2': ov2_n,
        }

        norm_feats = get_normalized_features(next_state_sim)
        vfa_k = next_weights['intercept'] + sum(
            next_weights[f] * norm_feats[f] for f in feature_cols
        )
        expected_vfa += (1.0 / K_SCENARIOS_BACKWARD) * vfa_k

    return imm_cost + expected_vfa


# ============================================================================
# MAIN LOOP: APPROXIMATE POLICY ITERATION
# ============================================================================
for i in range(ITERATIONS_I):
    print(f"\nOUTER LOOP i={i+1}/{ITERATIONS_I} (Policy Improvement)")

    visited_states_actions = {t: [] for t in range(T_HOURS)}
    current_states = []

    for n in range(N_SAMPLES):
        state_n = get_fixed_data().copy()
        state_n['T1']             = np.random.uniform(18.0, 26.0)
        state_n['T2']             = np.random.uniform(18.0, 26.0)
        state_n['H']              = np.random.uniform(20.0, 70.0)
        state_n['Occ1']           = np.random.uniform(25.0, 35.0)
        state_n['Occ2']           = np.random.uniform(15.0, 25.0)
        state_n['price_t']        = np.random.uniform(0.0,  12.0)
        state_n['price_previous'] = np.random.uniform(0.0,  12.0)
        state_n['current_time']   = 0
        current_states.append(state_n)

    # Forward pass
    for t in range(T_HOURS):
        next_t_weights = vfa_weights[t + 1] if t < T_HOURS - 1 else None
        for n in range(N_SAMPLES):
            state_n = current_states[n]
            state_n['current_time'] = t

            action = solve_bellman_equation_milp(state_n, next_t_weights)
            visited_states_actions[t].append((state_n.copy(), action))

            next_state_n, _ = apply_dynamics(state_n, action)
            if t + 1 < T_HOURS:
                new_occ1, new_occ2           = next_occupancy_levels(state_n['Occ1'], state_n['Occ2'])
                new_price                    = price_model(state_n['price_t'], state_n['price_previous'])
                next_state_n['Occ1']         = new_occ1
                next_state_n['Occ2']         = new_occ2
                next_state_n['price_previous']= state_n['price_t']
                next_state_n['price_t']      = new_price
            current_states[n] = next_state_n

    # Backward pass
    inner_weights = {t: dict(vfa_weights[t]) for t in range(T_HOURS)}

    for j in range(SWEEPS_J):
        print(f"  Inner Sweep j={j+1}/{SWEEPS_J} (Policy Evaluation)")
        for t in reversed(range(T_HOURS)):
            X_features, Y_targets = [], []
            next_t_weights_j = inner_weights[t + 1] if t < T_HOURS - 1 else None

            for state_n, action_n in visited_states_actions[t]:
                Y_targets.append(evaluate_fixed_action(state_n, action_n, next_t_weights_j))
                norm_feats = get_normalized_features(state_n)
                X_features.append([norm_feats[feat] for feat in feature_cols])

            if X_features:
                regressor = Ridge(alpha=1.0, fit_intercept=True)
                regressor.fit(X_features, Y_targets)
                for idx, feat_name in enumerate(feature_cols):
                    inner_weights[t][feat_name] = regressor.coef_[idx]
                inner_weights[t]['intercept'] = regressor.intercept_

    # Policy improvement
    for t in range(T_HOURS):
        for k in feature_cols + ['intercept']:
            vfa_weights[t][k] = (1 - BETA) * vfa_weights[t][k] + BETA * inner_weights[t][k]


# FINAL OUTPUT
print("\n=== FINAL RESULT: ADP_WEIGHTS ===")
for t in range(T_HOURS):
    clean_weights = {k: round(float(v), 4) for k, v in vfa_weights[t].items()}
    print(f"    {t}: {clean_weights},")

# Save to ADP_weights.csv
output_csv = Path(__file__).resolve().parent / "Data" / "ADP_weights.csv"
output_csv.parent.mkdir(parents=True, exist_ok=True)

csv_columns = ["t"] + feature_cols + ["intercept"]
with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_columns)
    writer.writeheader()
    for t in range(T_HOURS):
        row = {"t": t}
        for feat in feature_cols:
            row[feat] = float(vfa_weights[t][feat])
        row["intercept"] = float(vfa_weights[t]["intercept"])
        writer.writerow(row)

print(f"Saved ADP weights to CSV: {output_csv}")
