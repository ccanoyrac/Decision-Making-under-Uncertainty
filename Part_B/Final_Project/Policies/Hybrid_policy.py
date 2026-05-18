"""
Hybrid_policy.py
SP + Cost Function Approximation (CFA) Policy — Group 14, DTU 02435, Spring 2026.

Architecture
------------
Combines two complementary approaches:

  1. Short-horizon stochastic programming MILP on a k-means scenario tree
     (identical tree construction to SP_policy_14.py).  With HORIZON=1 this
     is a two-stage SP: here-and-now root decision + one recourse stage.

  2. Linear Value Function Approximation (VFA) trained offline by
     ADP_Training_NEW.py (weights stored in api_vfa_weights.json) added as a
     terminal cost at every leaf node of the tree.  This lets a shallow tree
     implicitly account for the remaining time steps without the exponential
     growth of a deeper scenario tree.

Objective (solved inside the Gurobi MILP):
    min  sum_{nodes} prob * price_node * (p1 + p2 + P_vent * v)   [SP cost]
       + LAMBDA_CFA * sum_{leaves} prob * VFA_{t+H+1}(leaf)       [CFA term]

VFA at leaf nid, using weights w = _VFA[t_now + HORIZON + 1]:
    VFA = w['T1']            * (T1[nid] - 22) / 8
        + w['T2']            * (T2[nid] - 22) / 8
        + w['H']             * (H[nid]  - 40) / 40
        + w['c']             * c_feat(v[nid])              [linear in v]
        + w['price_previous']* leaf_price / 10             [scalar]
        + E[w['price']*p/10 + w['occ1']*(o1-20)/30        [MC constant]
            + w['occ2']*(o2-10)/20]
        + w['intercept']

c_feat approximation (keeps the MILP linear):
    c_root >= 2  →  constant max(0, c_root-2)/3
    c_root < 2   →  v[nid] * (2/3)   [over-estimates by ≤1/3 when root also ON]

Public interface:
    select_action(state) -> {'HeatPowerRoom1', 'HeatPowerRoom2', 'VentilationON'}
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
import pyomo.environ as pyo
from Data.v2_SystemCharacteristics import get_fixed_data

# ── System parameters ─────────────────────────────────────────────────────────
_SYS = get_fixed_data()
_P = {
    'P_max'    : _SYS['heating_max_power'],
    'P_vent'   : _SYS['ventilation_power'],
    'T_low'    : _SYS['temp_min_comfort_threshold'],
    'T_OK'     : _SYS['temp_OK_threshold'],
    'T_high'   : _SYS['temp_max_comfort_threshold'],
    'H_high'   : _SYS['humidity_threshold'],
    'zeta_exch': _SYS['heat_exchange_coeff'],
    'zeta_loss': _SYS['thermal_loss_coeff'],
    'zeta_conv': _SYS['heating_efficiency_coeff'],
    'zeta_cool': _SYS['heat_vent_coeff'],
    'zeta_occ' : _SYS['heat_occupancy_coeff'],
    'eta_occ'  : _SYS['humidity_occupancy_coeff'],
    'eta_vent' : _SYS['humidity_vent_coeff'],
    'T_out'    : _SYS['outdoor_temperature'],
}
_T_OUT   = _SYS['outdoor_temperature']
_VENT_UP = 3

# ── VFA weights (trained by ADP_Training_NEW.py → api_vfa_weights.json) ──────
# vfa_weights[t] ≈ V_t(s) = expected cost-to-go from state s at step t.
# At a leaf node covering time t_leaf, the CFA weight index is t_leaf + 1.
_dir      = os.path.dirname(os.path.abspath(__file__))
_VFA_PATH = os.path.join(_dir, 'api_vfa_weights.json')
try:
    with open(_VFA_PATH) as _f:
        _vfa_raw = json.load(_f)
    _VFA = {int(k): v for k, v in _vfa_raw.items()}
except FileNotFoundError:
    _VFA = {}

# ── CSV scenario data ─────────────────────────────────────────────────────────
_price_df   = pd.read_csv(os.path.join(_dir, 'Data/v2_PriceData.csv'),   header=0)
_occ1_df    = pd.read_csv(os.path.join(_dir, 'Data/OccupancyRoom1.csv'), header=0)
_occ2_df    = pd.read_csv(os.path.join(_dir, 'Data/OccupancyRoom2.csv'), header=0)
_PRICES_ARR = _price_df[[str(i) for i in range(1, 11)]].values   # (n_days, 10)
_OCC1_ARR   = _occ1_df[[str(i)  for i in range(10)]].values      # (n_days, 10)
_OCC2_ARR   = _occ2_df[[str(i)  for i in range(10)]].values      # (n_days, 10)
_N_DAYS     = len(_PRICES_ARR)


# ── Private helpers ───────────────────────────────────────────────────────────

def _t_out(t: int) -> float:
    return float(_T_OUT[max(0, min(t, len(_T_OUT) - 1))])


def _sample_price_csv(t_slot: int, n: int) -> np.ndarray:
    col = min(max(t_slot, 0), 9)
    return np.random.choice(_PRICES_ARR[:, col], size=n, replace=True)


def _sample_occ_csv(t_slot: int, n: int):
    col = min(max(t_slot, 0), 9)
    idx = np.random.randint(0, _N_DAYS, size=n)
    return _OCC1_ARR[idx, col], _OCC2_ARR[idx, col]


def _cluster(features: np.ndarray, k: int):
    n = len(features)
    if n < k:
        features = np.tile(features, (int(np.ceil(k / n)), 1))[:k]
    std = features.std(axis=0)
    std[std == 0] = 1.0
    centers_n, labels = kmeans2(features / std, k, iter=10, minit='points', seed=42)
    centers = centers_n * std
    counts  = np.maximum(np.bincount(labels, minlength=k).astype(float), 1.0)
    return centers, counts / counts.sum()


def _build_tree(state: dict, horizon: int, branches: int, n_init: int) -> list:
    """Monte Carlo + k-means scenario tree (identical to SP_policy_14.py)."""
    t_now = int(state.get('current_time', 0))
    nodes = [{
        'id': 0, 'stage': 0, 'parent_id': None, 'children': [],
        'price'     : float(state['price_t']),
        'price_prev': float(state.get('price_previous', state['price_t'])),
        'occ1'      : float(state['Occ1']),
        'occ2'      : float(state['Occ2']),
        'T_out'     : _t_out(t_now),
        'prob'      : 1.0,
    }]
    m_cond   = max(n_init // branches, 30)
    leaf_ids = [0]

    for stage in range(horizon):
        n_smp        = n_init if stage == 0 else m_cond
        new_leaf_ids = []
        t_slot       = t_now + stage + 1
        for pid in leaf_ids:
            par          = nodes[pid]
            p_s          = _sample_price_csv(t_slot, n_smp)
            o1_s, o2_s   = _sample_occ_csv(t_slot, n_smp)
            centers, prb = _cluster(np.column_stack([p_s, o1_s, o2_s]), branches)
            for b in range(branches):
                cid = len(nodes)
                nodes.append({
                    'id'       : cid,
                    'stage'    : stage + 1,
                    'parent_id': pid,
                    'children' : [],
                    'price'    : float(centers[b, 0]),
                    'price_prev': par['price'],
                    'occ1'     : float(centers[b, 1]),
                    'occ2'     : float(centers[b, 2]),
                    'T_out'    : _t_out(t_slot),
                    'prob'     : par['prob'] * float(prb[b]),
                })
                par['children'].append(cid)
                new_leaf_ids.append(cid)
        leaf_ids = new_leaf_ids
    return nodes


def _cfa_expr(nid: int, nd: dict, m, w: dict, c_root: int,
               t_next: int, k_stoch: int = 40):
    """
    Return a Pyomo expression for the VFA terminal cost at leaf node `nid`.

    Parameters
    ----------
    nid    : leaf node id
    nd     : leaf node dict (has 'price', 'occ1', 'occ2', 'T_out')
    m      : Pyomo ConcreteModel (has m.T1, m.T2, m.H, m.v indexed by node id)
    w      : VFA weight dict for time t_next (keys: T1,T2,H,c,price,
             price_previous,occ1,occ2,intercept)
    c_root : vent_counter from the current observed state (int)
    t_next : time slot for VFA lookup (= t_now + HORIZON + 1), already clamped
    k_stoch: MC draws for the stochastic (price, occ) constant
    """
    # ── Deterministic VFA terms (linear Pyomo expressions) ────────────────────
    T1_feat = (m.T1[nid] - 22.0) / 8.0
    T2_feat = (m.T2[nid] - 22.0) / 8.0
    H_feat  = (m.H[nid]  - 40.0) / 40.0

    # c after leaf action (approximate linearly to keep MILP LP-relaxation valid)
    #   c_root >= 2: both root and leaf are forced ON; post-leaf c = max(0, c_root-2)
    #   c_root <  2: linear in v[nid]; approximates "start fresh" (c_post ≈ 2*v)
    if c_root >= 2:
        c_feat = float(max(0, c_root - 2)) / 3.0
    else:
        c_feat = m.v[nid] * (2.0 / 3.0)

    # leaf's own price becomes price_previous in the next period (deterministic)
    price_prev_feat = nd['price'] / 10.0

    det_expr = (
        w['T1']            * T1_feat
        + w['T2']          * T2_feat
        + w['H']           * H_feat
        + w['c']           * c_feat
        + w['price_previous'] * price_prev_feat
    )

    # ── Stochastic constant: E[VFA contribution from future price and occ] ────
    p_smp          = _sample_price_csv(t_next, k_stoch)
    o1_smp, o2_smp = _sample_occ_csv(t_next, k_stoch)
    stoch_const = float(np.mean(
        w['price'] * p_smp / 10.0
        + w['occ1'] * (o1_smp - 20.0) / 30.0
        + w['occ2'] * (o2_smp - 10.0) / 20.0
    ))

    return det_expr + stoch_const + w['intercept']


def _solve_hybrid(state: dict, nodes: list, lambda_cfa: float) -> dict:
    """
    Build and solve the hybrid SP + CFA MILP.

    The constraint structure (dynamics, overrule logic, min-up-time) is
    identical to SP_policy_14.py.  The objective extends the pure expected
    SP cost with a weighted CFA terminal value at every leaf node.
    """
    p       = _P
    T1_obs  = float(state['T1'])
    T2_obs  = float(state['T2'])
    H_obs   = float(state['H'])
    occ1_0  = float(state['Occ1'])
    occ2_0  = float(state['Occ2'])
    c       = int(state.get('vent_counter', 0))
    y_lo1_0 = int(state.get('low_override_r1', 0))
    y_lo2_0 = int(state.get('low_override_r2', 0))
    t_now   = int(state.get('current_time', 0))

    y_hi1_0 = 1 if T1_obs > p['T_high'] else 0
    y_hi2_0 = 1 if T2_obs > p['T_high'] else 0
    h_ov_0  = 1 if H_obs  > p['H_high'] else 0

    M_T, M_H  = 60.0, 200.0
    L         = _VENT_UP

    ids       = [n['id'] for n in nodes]
    nmap      = {n['id']: n for n in nodes}
    max_stage = max(n['stage'] for n in nodes)

    # VFA weight index for leaf nodes: cost-to-go starting one step after the
    # leaf's action (t_now + max_stage + 1), clamped to available range.
    vfa_t_idx = min(t_now + max_stage + 1, max(_VFA.keys(), default=9))
    w_leaf    = _VFA.get(vfa_t_idx, {})

    def _desc(nid, max_d):
        result, stack = [], [(nid, 0)]
        while stack:
            curr, d = stack.pop()
            if d > 0:
                result.append(curr)
            if d < max_d:
                for cid in nmap[curr]['children']:
                    stack.append((cid, d + 1))
        return result

    m = pyo.ConcreteModel()

    # ── Decision variables ─────────────────────────────────────────────────────
    m.p1 = pyo.Var(ids, bounds=(0.0, p['P_max']))
    m.p2 = pyo.Var(ids, bounds=(0.0, p['P_max']))
    m.v  = pyo.Var(ids, domain=pyo.Binary)

    # ── State variables ────────────────────────────────────────────────────────
    m.T1 = pyo.Var(ids, bounds=(-50.0, 100.0))
    m.T2 = pyo.Var(ids, bounds=(-50.0, 100.0))
    m.H  = pyo.Var(ids, bounds=(-200.0, 500.0))

    # ── Big-M overrule indicators ──────────────────────────────────────────────
    m.y_lo1  = pyo.Var(ids, domain=pyo.Binary)
    m.y_lo2  = pyo.Var(ids, domain=pyo.Binary)
    m.z_blo1 = pyo.Var(ids, domain=pyo.Binary)
    m.z_blo2 = pyo.Var(ids, domain=pyo.Binary)
    m.z_bok1 = pyo.Var(ids, domain=pyo.Binary)
    m.z_bok2 = pyo.Var(ids, domain=pyo.Binary)
    m.w1     = pyo.Var(ids, domain=pyo.Binary)
    m.w2     = pyo.Var(ids, domain=pyo.Binary)
    m.z_ahi1 = pyo.Var(ids, domain=pyo.Binary)
    m.z_ahi2 = pyo.Var(ids, domain=pyo.Binary)
    m.z_hum  = pyo.Var(ids, domain=pyo.Binary)

    m.cons = pyo.ConstraintList()

    # Fix initial overrule flags
    m.cons.add(m.y_lo1[0] == y_lo1_0)
    m.cons.add(m.y_lo2[0] == y_lo2_0)

    # Root dynamics
    To0 = nmap[0]['T_out']
    m.cons.add(m.T1[0] == T1_obs + p['zeta_exch']*(T2_obs-T1_obs) + p['zeta_loss']*(To0-T1_obs) + p['zeta_conv']*m.p1[0] - p['zeta_cool']*m.v[0] + p['zeta_occ']*occ1_0)
    m.cons.add(m.T2[0] == T2_obs + p['zeta_exch']*(T1_obs-T2_obs) + p['zeta_loss']*(To0-T2_obs) + p['zeta_conv']*m.p2[0] - p['zeta_cool']*m.v[0] + p['zeta_occ']*occ2_0)
    m.cons.add(m.H[0]  == H_obs  + p['eta_occ']*(occ1_0+occ2_0)  - p['eta_vent']*m.v[0])

    # Root overrule hard constraints
    if c > 0:   m.cons.add(m.v[0]  == 1)
    if h_ov_0:  m.cons.add(m.v[0]  == 1)
    if y_hi1_0: m.cons.add(m.p1[0] == 0.0)
    if y_hi2_0: m.cons.add(m.p2[0] == 0.0)

    # ── Per-node constraints (same as SP_policy_14.py) ─────────────────────────
    for nd in nodes:
        nid = nd['id']
        pid = nd['parent_id']
        s   = nd['stage']

        m.cons.add(m.p1[nid] >= p['P_max'] * m.y_lo1[nid])
        m.cons.add(m.p2[nid] >= p['P_max'] * m.y_lo2[nid])

        if pid is not None:
            m.cons.add(m.p1[nid] <= p['P_max'] * (1 - m.z_ahi1[pid]))
            m.cons.add(m.p2[nid] <= p['P_max'] * (1 - m.z_ahi2[pid]))
            m.cons.add(m.v[nid]  >= m.z_hum[pid])

        if pid is not None:
            To = nd['T_out']
            m.cons.add(m.T1[nid] == m.T1[pid] + p['zeta_exch']*(m.T2[pid]-m.T1[pid]) + p['zeta_loss']*(To-m.T1[pid]) + p['zeta_conv']*m.p1[nid] - p['zeta_cool']*m.v[nid] + p['zeta_occ']*nd['occ1'])
            m.cons.add(m.T2[nid] == m.T2[pid] + p['zeta_exch']*(m.T1[pid]-m.T2[pid]) + p['zeta_loss']*(To-m.T2[pid]) + p['zeta_conv']*m.p2[nid] - p['zeta_cool']*m.v[nid] + p['zeta_occ']*nd['occ2'])
            m.cons.add(m.H[nid]  == m.H[pid]  + p['eta_occ']*(nd['occ1']+nd['occ2']) - p['eta_vent']*m.v[nid])

        # Big-M indicator definitions
        m.cons.add(m.T1[nid] >= p['T_low']  - M_T *  m.z_blo1[nid])
        m.cons.add(m.T1[nid] <= p['T_low']  + M_T * (1 - m.z_blo1[nid]))
        m.cons.add(m.T2[nid] >= p['T_low']  - M_T *  m.z_blo2[nid])
        m.cons.add(m.T2[nid] <= p['T_low']  + M_T * (1 - m.z_blo2[nid]))
        m.cons.add(m.T1[nid] >= p['T_OK']   - M_T * (1 - m.z_bok1[nid]))
        m.cons.add(m.T1[nid] <= p['T_OK']   + M_T *    m.z_bok1[nid])
        m.cons.add(m.T2[nid] >= p['T_OK']   - M_T * (1 - m.z_bok2[nid]))
        m.cons.add(m.T2[nid] <= p['T_OK']   + M_T *    m.z_bok2[nid])
        m.cons.add(m.T1[nid] <= p['T_high'] + M_T *  m.z_ahi1[nid])
        m.cons.add(m.T1[nid] >= p['T_high'] - M_T * (1 - m.z_ahi1[nid]))
        m.cons.add(m.T2[nid] <= p['T_high'] + M_T *  m.z_ahi2[nid])
        m.cons.add(m.T2[nid] >= p['T_high'] - M_T * (1 - m.z_ahi2[nid]))
        m.cons.add(m.H[nid]  <= p['H_high'] + M_H *  m.z_hum[nid])

        if pid is not None:
            m.cons.add(m.y_lo1[nid] <= m.z_bok1[pid])
            m.cons.add(m.y_lo2[nid] <= m.z_bok2[pid])

        if nd['children']:
            m.cons.add(m.w1[nid] <= m.y_lo1[nid])
            m.cons.add(m.w1[nid] <= m.z_bok1[nid])
            m.cons.add(m.w1[nid] >= m.y_lo1[nid] + m.z_bok1[nid] - 1)
            m.cons.add(m.w2[nid] <= m.y_lo2[nid])
            m.cons.add(m.w2[nid] <= m.z_bok2[nid])
            m.cons.add(m.w2[nid] >= m.y_lo2[nid] + m.z_bok2[nid] - 1)
            for cid in nd['children']:
                m.cons.add(m.y_lo1[cid] >= m.z_blo1[nid])
                m.cons.add(m.y_lo1[cid] >= m.w1[nid])
                m.cons.add(m.y_lo1[cid] <= m.z_blo1[nid] + m.w1[nid])
                m.cons.add(m.y_lo2[cid] >= m.z_blo2[nid])
                m.cons.add(m.y_lo2[cid] >= m.w2[nid])
                m.cons.add(m.y_lo2[cid] <= m.z_blo2[nid] + m.w2[nid])

        v_prev_nd = (1 if c > 0 else 0) if pid is None else m.v[pid]
        max_d     = min(L - 1, max_stage - s)
        for did in _desc(nid, max_d):
            m.cons.add(m.v[did] >= m.v[nid] - v_prev_nd)

    # Ventilation inertia carry-over into future stages
    for nd in nodes:
        if nd['parent_id'] is not None and c > nd['stage']:
            m.cons.add(m.v[nd['id']] == 1)

    # ── Objective: SP cost + weighted CFA terminal at leaves ───────────────────
    sp_cost = sum(
        nd['prob'] * nd['price'] * (m.p1[nd['id']] + m.p2[nd['id']] + p['P_vent'] * m.v[nd['id']])
        for nd in nodes
    )

    cfa_cost = 0.0
    if lambda_cfa > 0.0 and w_leaf:
        t_next_slot = min(t_now + max_stage + 1, 9)
        cfa_cost = lambda_cfa * sum(
            nd['prob'] * _cfa_expr(
                nd['id'], nd, m, w_leaf, c,
                t_next=t_next_slot, k_stoch=40,
            )
            for nd in nodes if not nd['children']   # leaf nodes only
        )

    m.obj = pyo.Objective(expr=sp_cost + cfa_cost, sense=pyo.minimize)

    solver = pyo.SolverFactory('gurobi')
    solver.options['OutputFlag'] = 0
    solver.options['TimeLimit']  = 10
    res = solver.solve(m)

    ok = (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible)
    if res.solver.termination_condition in ok:
        return {
            'HeatPowerRoom1': float(pyo.value(m.p1[0])),
            'HeatPowerRoom2': float(pyo.value(m.p2[0])),
            'VentilationON' : int(round(float(pyo.value(m.v[0])))),
        }

    # Fallback: reactive
    return {
        'HeatPowerRoom1': _P['P_max'] if y_lo1_0 else 0.0,
        'HeatPowerRoom2': _P['P_max'] if y_lo2_0 else 0.0,
        'VentilationON' : 1 if (c > 0 or H_obs > _P['H_high']) else 0,
    }


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

def select_action(state: dict) -> dict:
    """
    Hybrid SP + CFA policy.

    Tunable parameters
    ------------------
    HORIZON    : int   SP look-ahead stages (1 = two-stage SP).
    BRANCHES   : int   k-means scenario clusters per stage.
    N_INIT     : int   Monte Carlo draws before clustering.
    LAMBDA_CFA : float Weight on the VFA terminal cost (0 = pure SP, 1 = full hybrid).

    Parameters
    ----------
    state : dict
        Keys: T1, T2, H, Occ1, Occ2, price_t, price_previous,
              vent_counter, low_override_r1, low_override_r2, current_time.

    Returns
    -------
    dict with keys HeatPowerRoom1, HeatPowerRoom2, VentilationON.
    """
    HORIZON    = 1      # two-stage: root + one recourse stage
    BRANCHES   = 10     # k-means clusters per stage
    N_INIT     = 1000   # MC draws for first-stage clustering
    LAMBDA_CFA = 1.0    # full weighting of VFA terminal cost

    p   = _P
    t   = int(state.get('current_time', 0))
    c   = int(state.get('vent_counter', 0))
    H   = float(state.get('H', 0.0))

    # Last time step: no future to plan for
    if t >= 9:
        return {
            'HeatPowerRoom1': p['P_max'] if state.get('low_override_r1') else 0.0,
            'HeatPowerRoom2': p['P_max'] if state.get('low_override_r2') else 0.0,
            'VentilationON' : 1 if (c > 0 or H >= p['H_high']) else 0,
        }

    # When no VFA weights are available, degrade gracefully to pure SP
    effective_lambda = LAMBDA_CFA if _VFA else 0.0

    try:
        nodes = _build_tree(state, HORIZON, BRANCHES, N_INIT)
        return _solve_hybrid(state, nodes, effective_lambda)
    except Exception:
        return {
            'HeatPowerRoom1': p['P_max'] if state.get('low_override_r1') else 0.0,
            'HeatPowerRoom2': p['P_max'] if state.get('low_override_r2') else 0.0,
            'VentilationON' : 1 if (c > 0 or H >= p['H_high']) else 0,
        }
