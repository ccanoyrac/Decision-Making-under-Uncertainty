"""Standalone ADP training script — mirrors ADP_policy_14.ipynb cells 01-08.

Run via:  conda run -n 02435_DMUU python train_adp.py
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import v2_SystemCharacteristics as sc

raw = sc.get_fixed_data()
params = {
    'P_max'    : raw['heating_max_power'],
    'P_vent'   : raw['ventilation_power'],
    'T_low'    : raw['temp_min_comfort_threshold'],
    'T_ok'     : raw['temp_OK_threshold'],
    'T_high'   : raw['temp_max_comfort_threshold'],
    'H_high'   : raw['humidity_threshold'],
    'zeta_exch': raw['heat_exchange_coeff'],
    'zeta_loss': raw['thermal_loss_coeff'],
    'zeta_conv': raw['heating_efficiency_coeff'],
    'zeta_cool': raw['heat_vent_coeff'],
    'zeta_occ' : raw['heat_occupancy_coeff'],
    'eta_occ'  : raw['humidity_occupancy_coeff'],
    'eta_vent' : raw['humidity_vent_coeff'],
    'T_out'    : raw['outdoor_temperature'],
}
T = 10
print("System parameters loaded.")
print(f"  T_low={params['T_low']} T_ok={params['T_ok']} T_high={params['T_high']} "
      f"H_high={params['H_high']} P_max={params['P_max']} P_vent={params['P_vent']}")


def _price_model(current_price, previous_price):
    next_p = (current_price + 0.6*(current_price - previous_price)
              + 0.12*(4.0 - current_price) + np.random.normal(0, 0.5))
    if next_p < 0 and np.random.rand() > 0.2:
        next_p = np.random.uniform(0, 1.2)
    return float(np.clip(next_p, 0.0, 12.0))


def _next_occupancy(r1, r2):
    r1n = r1 + 0.25*(35.0 - r1) + 0.1*(r2 - r1) + np.random.normal(0, 3.0)
    r2n = r2 + 0.25*(25.0 - r2) + 0.1*(r1 - r2) + np.random.normal(0, 2.5)
    return float(np.clip(r1n, 20, 50)), float(np.clip(r2n, 10, 30))


# ── Load data ──────────────────────────────────────────────────────────────────
price_df = pd.read_csv('v2_PriceData.csv', header=0)
occ1_df  = pd.read_csv('OccupancyRoom1.csv', header=0)
occ2_df  = pd.read_csv('OccupancyRoom2.csv', header=0)

price_cols  = [str(i) for i in range(1, 11)]
occ_cols    = [str(i) for i in range(10)]
prices_all  = price_df[price_cols].values.astype(float)
prev_prices = price_df.iloc[:, 0].values.astype(float)
occ1_all    = occ1_df[occ_cols].values.astype(float)
occ2_all    = occ2_df[occ_cols].values.astype(float)
N = len(prices_all)
print(f"Loaded {N} days. Price [{prices_all.min():.2f}, {prices_all.max():.2f}]")


# ── Feature mapping ────────────────────────────────────────────────────────────
N_FEAT = 12
K_MC   = 50


def phi(state):
    T1         = float(state['T1'])
    T2         = float(state['T2'])
    H          = float(state.get('H', 0.0))
    price      = float(state['price'])
    price_prev = float(state.get('price_previous', state['price']))
    occ1       = float(state.get('occ1', 0.0))
    occ2       = float(state.get('occ2', 0.0))
    c          = float(state.get('c', 0))
    pen1       = max(0.0, 22.0 - T1) / 3.0
    pen2       = max(0.0, 22.0 - T2) / 3.0
    p_norm     = price / 10.0
    return np.array([
        (T1   - 22.0) /  8.0,
        (T2   - 22.0) /  8.0,
        (H    - 40.0) / 40.0,
         p_norm,
         price_prev / 10.0,
        (occ1 - 30.0) / 30.0,
        (occ2 - 20.0) / 20.0,
         c    /  3.0,
         pen1,
         pen2,
         p_norm * pen1,
         p_norm * pen2,
    ], dtype=float)


# ── Dynamics + heuristic ───────────────────────────────────────────────────────
def apply_dynamics(state, decisions):
    p  = params
    t  = int(state['current_time'])
    p1 = float(decisions['HeatPowerRoom1'])
    p2 = float(decisions['HeatPowerRoom2'])
    v  = int(float(decisions['VentilationON']) > 0.5)
    H  = float(state.get('H', 0.0))
    c  = int(state.get('c', 0))

    if c > 0:                 v  = 1
    if H >= p['H_high']:      v  = 1
    if state.get('y_low_1'):  p1 = p['P_max']
    if state.get('y_low_2'):  p2 = p['P_max']
    if state.get('y_high_1'): p1 = 0.0
    if state.get('y_high_2'): p2 = 0.0

    step_cost = float(state['price']) * (p1 + p2 + p['P_vent'] * v)
    T_out = float(p['T_out'][min(t, 9)])
    T1 = float(state['T1']); T2 = float(state['T2'])
    occ1 = float(state.get('occ1', 0.0)); occ2 = float(state.get('occ2', 0.0))

    T1_n = (T1 + p['zeta_exch']*(T2-T1) + p['zeta_loss']*(T_out-T1)
            + p['zeta_conv']*p1 - p['zeta_cool']*v + p['zeta_occ']*occ1)
    T2_n = (T2 + p['zeta_exch']*(T1-T2) + p['zeta_loss']*(T_out-T2)
            + p['zeta_conv']*p2 - p['zeta_cool']*v + p['zeta_occ']*occ2)
    H_n  = max(0.0, H + p['eta_occ']*(occ1+occ2) - p['eta_vent']*v)
    c_n  = (2 if c == 0 else max(0, c-1)) if v == 1 else 0

    y_lo1 = (1 if T1_n < p['T_low'] else (0 if T1_n >= p['T_ok'] else int(state.get('y_low_1', 0))))
    y_lo2 = (1 if T2_n < p['T_low'] else (0 if T2_n >= p['T_ok'] else int(state.get('y_low_2', 0))))
    y_hi1 = 1 if T1_n > p['T_high'] else 0
    y_hi2 = 1 if T2_n > p['T_high'] else 0

    return {
        'T1': T1_n, 'T2': T2_n, 'H': H_n, 'c': c_n,
        'y_low_1': y_lo1, 'y_low_2': y_lo2,
        'y_high_1': y_hi1, 'y_high_2': y_hi2,
        'occ1': occ1, 'occ2': occ2,
        'price': float(state['price']),
        'price_previous': float(state['price']),
        'current_time': t + 1,
    }, step_cost


def heuristic_policy(state):
    p = params
    T1 = float(state['T1']); T2 = float(state['T2'])
    H  = float(state.get('H', 0.0)); c = int(state.get('c', 0))
    v  = 1 if (c > 0 or H >= p['H_high']) else int(H > 55.0)

    def heat_power(T_r, y_lo, y_hi):
        if y_hi: return 0.0
        if y_lo: return p['P_max']
        return p['P_max'] if T_r < (p['T_ok'] - 1.0) else 0.0

    return {
        'HeatPowerRoom1': heat_power(T1, state.get('y_low_1', 0), state.get('y_high_1', 0)),
        'HeatPowerRoom2': heat_power(T2, state.get('y_low_2', 0), state.get('y_high_2', 0)),
        'VentilationON' : v,
    }


# ── Forward pass ───────────────────────────────────────────────────────────────
N_SAMPLES = 50


def run_forward_pass(policy_fn):
    states_out  = [[None] * N_SAMPLES for _ in range(T)]
    day_indices = np.random.choice(N, N_SAMPLES, replace=True)

    for n, di in enumerate(day_indices):
        T1_0    = np.random.uniform(17.5, 23.5)
        T2_0    = np.random.uniform(17.5, 23.5)
        H_0     = np.random.uniform(30.0, 60.0)
        c_0     = int(np.random.choice([0, 1, 2]))
        y_lo1_0 = int(np.random.choice([0, 1]))
        y_lo2_0 = int(np.random.choice([0, 1]))

        state = {
            'T1': T1_0, 'T2': T2_0, 'H': H_0,
            'c': c_0, 'y_low_1': y_lo1_0, 'y_low_2': y_lo2_0,
            'y_high_1': 0, 'y_high_2': 0,
            'occ1'          : float(occ1_all[di, 0]),
            'occ2'          : float(occ2_all[di, 0]),
            'price'         : float(prices_all[di, 0]),
            'price_previous': float(prev_prices[di]),
            'current_time'  : 0,
        }
        for t in range(T):
            state['price']          = float(prices_all[di, t])
            state['price_previous'] = (float(prices_all[di, t-1]) if t > 0
                                       else float(prev_prices[di]))
            state['occ1']           = float(occ1_all[di, t])
            state['occ2']           = float(occ2_all[di, t])
            state['current_time']   = t
            states_out[t][n] = dict(state)
            state, _ = apply_dynamics(state, policy_fn(state))

    return states_out


# ── One-step MILP + Bellman target ────────────────────────────────────────────
BIG_M = 1_000


def solve_1step(state, eta_next, return_action=False):
    p      = params
    P_max  = p['P_max']
    t      = int(state.get('current_time', 0))
    T_out  = float(p['T_out'][min(t, 9)])
    T1     = float(state['T1'])
    T2     = float(state['T2'])
    H      = float(state.get('H', 0.0))
    c      = int(state.get('c', 0))
    occ1   = float(state.get('occ1', 0.0))
    occ2   = float(state.get('occ2', 0.0))
    price  = float(state['price'])
    price_prev = float(state.get('price_previous', price))
    eta    = eta_next

    mc_prices = [_price_model(price, price_prev) for _ in range(K_MC)]
    mc_occs   = [_next_occupancy(occ1, occ2)     for _ in range(K_MC)]

    cond_price_next = float(np.mean(mc_prices))
    cond_occ1_next  = float(np.mean([o[0] for o in mc_occs]))
    cond_occ2_next  = float(np.mean([o[1] for o in mc_occs]))
    e_p_norm        = cond_price_next / 10.0

    m = pyo.ConcreteModel()
    m.p1  = pyo.Var(bounds=(0, P_max))
    m.p2  = pyo.Var(bounds=(0, P_max))
    m.v   = pyo.Var(domain=pyo.Binary)
    m.T1x = pyo.Var()
    m.T2x = pyo.Var()
    m.Hx  = pyo.Var()

    m.dT1 = pyo.Constraint(expr=
        m.T1x == T1 + p['zeta_exch']*(T2-T1) + p['zeta_loss']*(T_out-T1)
               + p['zeta_conv']*m.p1 - p['zeta_cool']*m.v + p['zeta_occ']*occ1)
    m.dT2 = pyo.Constraint(expr=
        m.T2x == T2 + p['zeta_exch']*(T1-T2) + p['zeta_loss']*(T_out-T2)
               + p['zeta_conv']*m.p2 - p['zeta_cool']*m.v + p['zeta_occ']*occ2)
    m.dH  = pyo.Constraint(expr=
        m.Hx  == H + p['eta_occ']*(occ1+occ2) - p['eta_vent']*m.v)

    if   c == 0: c_next_norm = (2.0/3.0) * m.v
    elif c == 1: c_next_norm = 0.0
    else:        c_next_norm = 1.0/3.0

    m.pen1_x = pyo.Var(bounds=(0.0, 30.0))
    m.pen2_x = pyo.Var(bounds=(0.0, 30.0))
    m.c_pen1 = pyo.Constraint(expr=m.pen1_x >= 22.0 - m.T1x)
    m.c_pen2 = pyo.Constraint(expr=m.pen2_x >= 22.0 - m.T2x)

    penalty_terms = []
    if c > 0 or H >= p['H_high']:
        penalty_terms.append(BIG_M * (1 - m.v))
    if state.get('y_low_1'):
        penalty_terms.append(BIG_M * (P_max - m.p1))
    if state.get('y_low_2'):
        penalty_terms.append(BIG_M * (P_max - m.p2))
    if state.get('y_high_1'):
        penalty_terms.append(BIG_M * m.p1)
    if state.get('y_high_2'):
        penalty_terms.append(BIG_M * m.p2)
    overrule_penalty = sum(penalty_terms) if penalty_terms else 0.0

    vfa_next = (
        eta[0]  * ((m.T1x          - 22.0) /  8.0) +
        eta[1]  * ((m.T2x          - 22.0) /  8.0) +
        eta[2]  * ((m.Hx           - 40.0) / 40.0) +
        eta[3]  * ( cond_price_next          / 10.0) +
        eta[4]  * ( price                    / 10.0) +
        eta[5]  * ((cond_occ1_next - 30.0) / 30.0) +
        eta[6]  * ((cond_occ2_next - 20.0) / 20.0) +
        eta[7]  *  c_next_norm                      +
        eta[8]  * (m.pen1_x / 3.0)                 +
        eta[9]  * (m.pen2_x / 3.0)                 +
        eta[10] * e_p_norm * (m.pen1_x / 3.0)      +
        eta[11] * e_p_norm * (m.pen2_x / 3.0)      +
        eta[12]
    )

    m.obj = pyo.Objective(
        expr=price*(m.p1 + m.p2 + p['P_vent']*m.v) + vfa_next + overrule_penalty,
        sense=pyo.minimize)

    solver = pyo.SolverFactory('gurobi')
    solver.options['OutputFlag'] = 0

    try:
        result = solver.solve(m)
    except Exception as exc:
        print(f"PYOMO CRASHED: {exc}")
        return None

    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        return None

    p1_val = float(pyo.value(m.p1))
    p2_val = float(pyo.value(m.p2))
    v_val  = int(round(pyo.value(m.v)))

    if return_action:
        return {'HeatPowerRoom1': p1_val, 'HeatPowerRoom2': p2_val, 'VentilationON': v_val}

    T1x_val = float(pyo.value(m.T1x))
    T2x_val = float(pyo.value(m.T2x))
    Hx_val  = float(pyo.value(m.Hx))
    c_next  = (2 if c == 0 else max(0, c-1)) if v_val == 1 else 0
    immediate = price * (p1_val + p2_val + p['P_vent'] * v_val)

    # Cascade penalty in Bellman targets only (not in online MILP)
    _T_low  = float(p['T_low'])
    _N_CASC = 3.0
    casc_penalty = 0.0
    if T1x_val < _T_low:
        casc_penalty += _N_CASC * cond_price_next * P_max * (_T_low - T1x_val)
    if T2x_val < _T_low:
        casc_penalty += _N_CASC * cond_price_next * P_max * (_T_low - T2x_val)

    vfa_acc = 0.0
    for k in range(K_MC):
        next_mc = {
            'T1': T1x_val, 'T2': T2x_val, 'H': Hx_val, 'c': c_next,
            'price'         : mc_prices[k],
            'price_previous': price,
            'occ1'          : mc_occs[k][0],
            'occ2'          : mc_occs[k][1],
        }
        vfa_acc += float(np.dot(phi(next_mc), eta[:N_FEAT]) + eta[N_FEAT])
    return immediate + vfa_acc / K_MC + casc_penalty


# ── Backward pass ──────────────────────────────────────────────────────────────
N_OUTER = 25
N_INNER = 5
BETA    = 0.25

eta_all = np.zeros((T + 1, N_FEAT + 1))


def run_backward_pass(states, verbose=False):
    eta_new = np.zeros((T + 1, N_FEAT + 1))

    for t in range(T - 1, -1, -1):
        eta_next = eta_new[t + 1]
        targets, rows, n_skip = [], [], 0

        for n in range(N_SAMPLES):
            v_star = solve_1step(states[t][n], eta_next)
            if v_star is None or not np.isfinite(v_star):
                n_skip += 1
                continue
            targets.append(v_star)
            rows.append(phi(states[t][n]))

        n_valid = len(targets)
        if n_valid < N_FEAT + 1:
            eta_new[t] = eta_all[t]
            if verbose:
                print(f"    t={t}  WARN: only {n_valid} valid samples — kept old weights")
            continue

        Phi   = np.array(rows)
        y     = np.array(targets)
        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(Phi, y)

        coefs = ridge.coef_.copy()
        for idx in (7, 8, 9, 10, 11):
            coefs[idx] = max(0.0, coefs[idx])
        eta_new[t] = np.append(coefs, ridge.intercept_)

        if verbose:
            y_hat = Phi @ coefs + ridge.intercept_
            rmse  = float(np.sqrt(np.mean((y_hat - y)**2)))
            print(f"    t={t}  n={n_valid}/{N_SAMPLES}  skip={n_skip}"
                  f"  rmse={rmse:8.3f}  |w|∞={np.max(np.abs(coefs)):.3f}")

    return eta_new


def adp_policy(state):
    t        = int(state.get('current_time', 0))
    eta_next = eta_all[min(t + 1, T)]
    action   = solve_1step(state, eta_next, return_action=True)
    return action if action is not None else heuristic_policy(state)


# ── Main training loop ─────────────────────────────────────────────────────────
print(f"\nStarting API: N_OUTER={N_OUTER}, N_INNER={N_INNER}, N_SAMPLES={N_SAMPLES}, BETA={BETA}")
t_total = time.perf_counter()

for outer in range(N_OUTER):
    policy_fn = heuristic_policy if outer == 0 else adp_policy
    print(f"\n{'='*68}")
    print(f"API ITER {outer+1:>2}/{N_OUTER}  ·  "
          f"{'heuristic (iter 0)' if outer == 0 else 'ADP greedy'}")
    print(f"{'='*68}")

    t_fwd  = time.perf_counter()
    states = run_forward_pass(policy_fn)
    print(f"  Forward done: {time.perf_counter()-t_fwd:.1f}s  ({N_SAMPLES} trajectories)")

    for inner in range(N_INNER):
        verbose  = (inner == N_INNER - 1)
        eta_raw  = run_backward_pass(states, verbose=verbose)
        eta_all  = (1.0 - BETA) * eta_all + BETA * eta_raw

        print(f"  Inner {inner+1}/{N_INNER}"
              f"  max|w|={np.max(np.abs(eta_all[:T, :N_FEAT])):.3f}"
              f"  |b|∞={np.max(np.abs(eta_all[:T, N_FEAT])):.3f}")

elapsed = (time.perf_counter() - t_total) / 60.0
print(f"\nAll {N_OUTER}×{N_INNER} sweeps complete — {elapsed:.1f} min total.")

# ── Save weights ───────────────────────────────────────────────────────────────
os.makedirs('output', exist_ok=True)
weights_path = 'output/adp_weights.json'

_nan_count = int(np.sum(~np.isfinite(eta_all)))
if _nan_count > 0:
    print(f"WARNING: {_nan_count} NaN/Inf values replaced with 0.0")
    eta_all = np.where(np.isfinite(eta_all), eta_all, 0.0)
else:
    print(f"Weights sanity check OK — all {eta_all.size} values finite.")

weights_dict = {
    'description': (
        f'Time-dependent linear VFA weights for t=0..{T-1}. '
        f'Produced by train_adp.py (API, N_OUTER={N_OUTER}, '
        f'N_INNER={N_INNER}, N_SAMPLES={N_SAMPLES}, BETA={BETA}, Ridge alpha=1.0). '
        f'12 normalised features + Ridge intercept at index 12. '
        f'Cascade penalty (N_CASC=3) in Bellman targets only.'
    ),
    'feature_names': [
        '(T1-22)/8', '(T2-22)/8', '(H-40)/40',
        'price/10', 'price_prev/10',
        '(occ1-30)/30', '(occ2-20)/20',
        'c/3',
        'max(0,22.0-T1)/3', 'max(0,22.0-T2)/3',
        '(price/10)*max(0,22.0-T1)/3', '(price/10)*max(0,22.0-T2)/3',
        'intercept',
    ],
    'eta': {str(t): eta_all[t].tolist() for t in range(T)},
}

with open(weights_path, 'w') as f:
    json.dump(weights_dict, f, indent=2)

print(f"\nWeights saved → '{weights_path}'")
print(f"\nFinal VFA weights (t=0..{T-1}):")
hdr = (f"{'t':>2}  {'T1':>7}  {'T2':>7}  {'H':>7}  {'pr':>7}  {'pp':>7}"
       f"  {'oc1':>7}  {'oc2':>7}  {'c':>7}  {'pen1':>7}  {'pen2':>7}"
       f"  {'p*pen1':>8}  {'p*pen2':>8}  {'intercept':>12}")
print(hdr)
print("-" * len(hdr))
for ti in range(T):
    e = eta_all[ti]
    print(f"  {ti}  {e[0]:>7.3f}  {e[1]:>7.3f}  {e[2]:>7.3f}  {e[3]:>7.3f}  {e[4]:>7.3f}"
          f"  {e[5]:>7.3f}  {e[6]:>7.3f}  {e[7]:>7.3f}  {e[8]:>7.3f}  {e[9]:>7.3f}"
          f"  {e[10]:>8.3f}  {e[11]:>8.3f}  {e[12]:>12.3f}")