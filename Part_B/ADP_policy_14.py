"""
ADP_policy_14.py — Online Approximate Dynamic Programming Policy

Implements a time-dependent linear Value Function Approximation (VFA):

    V_hat_t(x_t) = phi(x_t)^T * eta_t[:8] + eta_t[8]

where phi: R^state -> R^8 is a hand-crafted normalised feature mapping and
eta_t is an 8-weight vector plus one Ridge intercept (9 values total) trained
offline by the Approximate Policy Iteration (API) loop in ADP_policy_14.ipynb.

At each decision epoch t, the policy solves a single-period MILP under the
certainty-equivalence (CE) principle:

    min_{p1, p2, v}  price * (p1 + p2 + P_vent * v)
                   + V_hat_{t+1}( E[x_{t+1} | x_t, p1, p2, v] )

The MILP has one binary variable (v) and is solved in milliseconds by Gurobi.
CE is valid because V_hat is strictly linear in (T1, T2, H, c) and the
transition dynamics are linear in the exogenous (occ, price) variables.

Entry point called by Task6_Environment.run_policy:
    select_action(state: dict) -> dict
"""

import os
import json
import numpy as np
import pyomo.environ as pyo
import v2_SystemCharacteristics as sc

# ==============================================================================
# 1.  SYSTEM PARAMETERS
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
    'eta_vent'  : _raw['humidity_vent_coeff'],
    'T_out'    : _raw['outdoor_temperature'],
}

# ==============================================================================
# 2.  FEATURE MAPPING
#
# phi(x_t) in R^8 — each component is normalised to lie in approximately
# [-1, 1] under typical operating conditions, ensuring numerical stability
# of the Ridge regression and MILP coefficients.
#
#   idx  feature        formula               baseline  scale
#    0   T1             (T1   - 22) / 8       22 °C     8 °C
#    1   T2             (T2   - 22) / 8       22 °C     8 °C
#    2   H              (H    - 40) / 40       40 %      40 %
#    3   price           price / 10            0         10 €/kWh
#    4   price_prev      price_prev / 10       0         10 €/kWh
#    5   occ1           (occ1 - 30) / 30       30 pax    30
#    6   occ2           (occ2 - 20) / 20       20 pax    20
#    7   c               c / 3                 0         3
#
# The Ridge intercept is fitted separately (fit_intercept=True) and stored
# as the 9th component of eta[t]:  VFA = phi(x)^T eta[:8] + eta[8].
# ==============================================================================
N_FEAT = 8


def _phi(state: dict) -> np.ndarray:
    """
    Compute the 8-dimensional normalised feature vector for a given state.

    The intercept eta[8] is handled by Ridge(fit_intercept=True) during
    training and evaluated separately in select_action; it is NOT included
    in the returned array.

    Parameters
    ----------
    state : dict
        Must contain 'T1', 'T2', 'price'.  Keys 'H', 'price_previous',
        'occ1', 'occ2', 'c' default to 0 / current-price if absent.

    Returns
    -------
    np.ndarray of shape (8,)
    """
    T1         = float(state['T1'])
    T2         = float(state['T2'])
    H          = float(state.get('H', 0.0))
    price      = float(state['price'])
    price_prev = float(state.get('price_previous', price))
    occ1       = float(state.get('occ1', 0.0))
    occ2       = float(state.get('occ2', 0.0))
    c          = float(state.get('c', 0))

    return np.array([
        (T1         - 22.0) /  8.0,   # idx 0 — room-1 temperature
        (T2         - 22.0) /  8.0,   # idx 1 — room-2 temperature
        (H          - 40.0) / 40.0,   # idx 2 — relative humidity
         price               / 10.0,  # idx 3 — current electricity price
         price_prev          / 10.0,  # idx 4 — lagged price (AR signal)
        (occ1       - 30.0) / 30.0,   # idx 5 — room-1 occupancy
        (occ2       - 20.0) / 20.0,   # idx 6 — room-2 occupancy
         c                  /  3.0,   # idx 7 — ventilation inertia counter
    ], dtype=float)


# ==============================================================================
# 3.  LINEAR VFA WEIGHTS
#
# Produced offline by ADP_policy_14.ipynb (API training).
# Each eta[t] is a 9-vector: [w_0 .. w_7, intercept].
# VFA evaluation: phi(x) @ eta[:8] + eta[8].
# ==============================================================================
_HERE        = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS_PATH = os.path.join(_HERE, 'output', 'adp_weights.json')

try:
    with open(_WEIGHTS_PATH) as _f:
        _w = json.load(_f)
    ETA = {int(t): _w['eta'][t] for t in _w['eta']}

    # Replace any NaN / Inf entries that may arise from numerical instability
    # during training; zero is a safe neutral value.
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
    print(f"[ADP] WARNING: {_WEIGHTS_PATH} not found — using zero weights.")
    ETA = {t: [0.0] * (N_FEAT + 1) for t in range(10)}


# ==============================================================================
# 4.  STATE-CONDITIONAL EXPECTED NEXT-PERIOD VALUES
#
# The electricity price follows an AR(1) process with mean-reversion:
#     p_{t+1} = p_t + 0.6*(p_t - p_{t-1}) + 0.12*(4 - p_t) + N(0, 0.5)
# Using the global mean as a certainty-equivalent price (the classic CE
# substitution) underestimates future cost during high-price periods because
# the AR process retains momentum.  Drawing K=50 Monte Carlo samples from the
# actual conditional distribution yields an accurate scalar constant, preserving
# MILP linearity while correcting the bias.
# ==============================================================================
K = 50


def _price_model(current_price: float, previous_price: float) -> float:
    """One-step AR price process — mirrors PriceProcessRestaurant.price_model."""
    next_p = (current_price
              + 0.6 * (current_price - previous_price)
              + 0.12 * (4.0 - current_price)
              + float(np.random.normal(0, 0.5)))
    if next_p < 0 and np.random.rand() > 0.2:
        next_p = float(np.random.uniform(0, 1.2))
    return float(np.clip(next_p, 0.0, 12.0))


def _next_occupancy(r1: float, r2: float):
    """One-step Markov occupancy — mirrors OccupancyProcessRestaurant.next_occupancy_levels."""
    r1_next = r1 + 0.25 * (35.0 - r1) + 0.1 * (r2 - r1) + float(np.random.normal(0, 3.0))
    r2_next = r2 + 0.25 * (25.0 - r2) + 0.1 * (r1 - r2) + float(np.random.normal(0, 2.5))
    return float(np.clip(r1_next, 20, 50)), float(np.clip(r2_next, 10, 30))


def _expected_exogenous(price: float, price_prev: float,
                        occ1: float, occ2: float):
    """
    Return (E[occ1_next], E[occ2_next], E[price_next]) conditioned on the
    current state via K=50 Monte Carlo draws from the actual process models.
    """
    mc_prices = [_price_model(price, price_prev) for _ in range(K)]
    mc_o1, mc_o2 = [], []
    for _ in range(K):
        o1, o2 = _next_occupancy(occ1, occ2)
        mc_o1.append(o1)
        mc_o2.append(o2)
    return (float(np.mean(mc_o1)),
            float(np.mean(mc_o2)),
            float(np.mean(mc_prices)))


# ==============================================================================
# 5.  ONE-STEP MILP  (certainty-equivalence formulation)
# ==============================================================================
def solve_adp_step(state: dict, eta: list, params: dict) -> dict:
    """
    Solve the one-period lookahead MILP under the certainty-equivalence principle.

    The MILP minimises:

        price * (p1 + p2 + P_vent * v)  +  phi(E[x_{t+1}])^T * eta[:8]  +  eta[8]
        +  Big-M overrule penalties

    over continuous heating powers p1, p2 in [0, P_max] and binary
    ventilation decision v in {0, 1}.

    Post-decision state (T1x, T2x, Hx) is encoded as auxiliary Pyomo
    variables constrained by the deterministic transition dynamics.
    Features 0, 1, 2, 7 of phi(E[x_{t+1}]) are Pyomo affine expressions
    linear in (p1, p2, v); features 3, 4, 5, 6 are scalar constants
    computed before the MILP is assembled (CE substitution).

    Parameters
    ----------
    state  : current state dict (keys: T1, T2, H, c, occ1, occ2, price,
             price_previous, current_time, y_low_1, y_low_2, y_high_1, y_high_2)
    eta    : weight vector for the current timestep (length N_FEAT+1 = 9)
    params : physical parameter dict (loaded from v2_SystemCharacteristics)

    Returns
    -------
    dict with keys 'HeatPowerRoom1', 'HeatPowerRoom2', 'VentilationON'
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

    # Conditional expected next-period values from K=50 MC draws.
    price_prev = float(state.get('price_previous', price))
    exp_occ1_next, exp_occ2_next, exp_price_next = _expected_exogenous(
        price, price_prev, occ1, occ2)

    m = pyo.ConcreteModel()

    # ------------------------------------------------------------------
    # Decision variables
    # ------------------------------------------------------------------
    m.p1 = pyo.Var(bounds=(0, P_max))
    m.p2 = pyo.Var(bounds=(0, P_max))
    m.v  = pyo.Var(domain=pyo.Binary)

    # ------------------------------------------------------------------
    # Post-decision state variables (auxiliary — determined by dynamics)
    # ------------------------------------------------------------------
    m.T1x = pyo.Var()
    m.T2x = pyo.Var()
    m.Hx  = pyo.Var()

    # ------------------------------------------------------------------
    # Transition dynamics  (linear in p1, p2, v)
    # ------------------------------------------------------------------
    m.dT1 = pyo.Constraint(expr=
        m.T1x == T1 + p['zeta_exch'] * (T2 - T1)
               + p['zeta_loss'] * (T_out - T1)
               + p['zeta_conv'] * m.p1
               - p['zeta_cool'] * m.v
               + p['zeta_occ']  * occ1)

    m.dT2 = pyo.Constraint(expr=
        m.T2x == T2 + p['zeta_exch'] * (T1 - T2)
               + p['zeta_loss'] * (T_out - T2)
               + p['zeta_conv'] * m.p2
               - p['zeta_cool'] * m.v
               + p['zeta_occ']  * occ2)

    m.dH = pyo.Constraint(expr=
        m.Hx == H
              + p['eta_occ']  * (occ1 + occ2)
              - p['eta_vent'] * m.v)

    # ------------------------------------------------------------------
    # Ventilation inertia counter (linear in v)
    #
    # The counter c tracks mandatory ventilation cycles:
    #   c = 0  (free):   c_next = 2*v       => normalised c_next/3 = (2/3)*v
    #   c = 1  (forced): c_next = 0         => normalised c_next/3 = 0
    #   c = 2  (forced): c_next = 1         => normalised c_next/3 = 1/3
    # ------------------------------------------------------------------
    if   c == 0: c_next_norm = (2.0 / 3.0) * m.v
    elif c == 1: c_next_norm = 0.0
    else:        c_next_norm = 1.0 / 3.0

    # ------------------------------------------------------------------
    # VFA of the expected next state:
    #   phi(E[x_{t+1}])^T * eta[:8]  +  eta[8]
    #
    # Features 0-2 and 7 are Pyomo expressions (linear in p1, p2, v).
    # Features 3-6 are evaluated as scalar constants (CE substitution).
    # ------------------------------------------------------------------
    vfa_next = (
        eta[8]                                                   +  # Ridge intercept
        eta[0] * ((m.T1x          - 22.0) /  8.0)               +  # T1 feature
        eta[1] * ((m.T2x          - 22.0) /  8.0)               +  # T2 feature
        eta[2] * ((m.Hx           - 40.0) / 40.0)               +  # humidity
        eta[3] * ( exp_price_next          / 10.0)               +  # E[price_next]
        eta[4] * ( price                   / 10.0)               +  # price_prev signal
        eta[5] * ((exp_occ1_next  - 30.0) / 30.0)               +  # E[occ1_next]
        eta[6] * ((exp_occ2_next  - 20.0) / 20.0)               +  # E[occ2_next]
        eta[7] *  c_next_norm                                       # inertia counter
    )

    # ------------------------------------------------------------------
    # Big-M soft penalties for mandatory overrule actions
    # These terms dominate the objective so that the solver obeys the
    # overrule logic set by the environment, without hardcoding equality
    # constraints that would alter the feasible region for normal steps.
    # ------------------------------------------------------------------
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
        expr=price * (m.p1 + m.p2 + p['P_vent'] * m.v) + vfa_next + overrule_penalty,
        sense=pyo.minimize)

    solver = pyo.SolverFactory('gurobi')
    solver.options['OutputFlag'] = 0

    result = solver.solve(m)

    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(
            f"[ADP] Solver failed at t={t}: {result.solver.termination_condition}")

    return {
        'HeatPowerRoom1': float(pyo.value(m.p1)),
        'HeatPowerRoom2': float(pyo.value(m.p2)),
        'VentilationON' : int(round(pyo.value(m.v))),
    }


# ==============================================================================
# 6.  POLICY ENTRY POINT
#
# Called by Task6_Environment.run_policy at each decision epoch t = 0..9.
# ==============================================================================
def select_action(state: dict) -> dict:
    """
    Primary interface invoked by the evaluation environment.

    Loads the pre-trained weights for the current timestep, solves the
    one-period MILP, and returns the optimal control actions.

    Parameters
    ----------
    state : dict
        State dictionary supplied by Task6_Environment at each timestep.
        Required keys: 'T1', 'T2', 'price', 'current_time'.

    Returns
    -------
    dict with keys 'HeatPowerRoom1', 'HeatPowerRoom2', 'VentilationON'
    """
    t         = int(state.get('current_time', 0))
    eta       = ETA.get(t, ETA[max(ETA.keys())])
    decisions = solve_adp_step(state, eta, params)

    # Diagnostic log — one line per timestep for performance monitoring.
    p1    = decisions['HeatPowerRoom1']
    p2    = decisions['HeatPowerRoom2']
    v     = decisions['VentilationON']
    price = float(state['price'])
    imm   = price * (p1 + p2 + params['P_vent'] * v)
    vfa   = float(np.dot(_phi(state), eta[:N_FEAT]) + eta[N_FEAT])
    print(f"  t={t} | p1={p1:.2f} p2={p2:.2f} v={v}"
          f" | imm={imm:.3f} vfa={vfa:.2f}"
          f" | price={price:.2f} T1={state['T1']:.1f} T2={state['T2']:.1f}")

    return decisions