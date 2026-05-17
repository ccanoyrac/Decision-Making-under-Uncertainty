"""
SP policy 14.py
Multi-stage Stochastic Programming Policy — Group 14, DTU 02435, Spring 2026.

Public interface:
    select_action(state) -> {'HeatPowerRoom1', 'HeatPowerRoom2', 'VentilationON'}
"""

import os
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
import pyomo.environ as pyo
from v2_SystemCharacteristics import get_fixed_data

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
    "T_out"     : _SYS['outdoor_temperature'],
}
_T_OUT     = _SYS['outdoor_temperature']   # deterministic outdoor temp schedule (known in advance)
_VENT_UP   = 3                             # minimum ventilation on-time (hours)

# ── CSV-based scenario data ───────────────────────────────────────────────────
# Prices and occupancies are drawn empirically from the historical CSV files,
# matching the data source used by the hindsight optimisation.
_dir = os.path.dirname(os.path.abspath(__file__))
_price_df = pd.read_csv(os.path.join(_dir, 'v2_PriceData.csv'),    header=0)
_occ1_df  = pd.read_csv(os.path.join(_dir, 'OccupancyRoom1.csv'),  header=0)
_occ2_df  = pd.read_csv(os.path.join(_dir, 'OccupancyRoom2.csv'),  header=0)

# price columns '1'–'10' → 0-indexed slots 0–9
# occupancy columns '0'–'9' → 0-indexed slots 0–9
_PRICES_ARR = _price_df[[str(i) for i in range(1, 11)]].values   # (n_days, 10)
_OCC1_ARR   = _occ1_df[[str(i)  for i in range(10)]].values      # (n_days, 10)
_OCC2_ARR   = _occ2_df[[str(i)  for i in range(10)]].values      # (n_days, 10)
_N_DAYS     = len(_PRICES_ARR)


# ── Private helpers ───────────────────────────────────────────────────────────

def _t_out(t):
    return float(_T_OUT[max(0, min(t, len(_T_OUT) - 1))])


def _sample_price_csv(t_slot, n):
    """Sample n prices from the CSV at time slot t_slot (0-indexed, clamped to 0–9)."""
    col = min(max(t_slot, 0), 9)
    return np.random.choice(_PRICES_ARR[:, col], size=n, replace=True)


def _sample_occ_csv(t_slot, n):
    """Sample n correlated (occ1, occ2) pairs from the CSV at time slot t_slot (0-indexed)."""
    col = min(max(t_slot, 0), 9)
    idx = np.random.randint(0, _N_DAYS, size=n)
    return _OCC1_ARR[idx, col], _OCC2_ARR[idx, col]


def _cluster(features, k):
    n = len(features)
    if n < k:
        features = np.tile(features, (int(np.ceil(k / n)), 1))[:k]
    std = features.std(axis=0); std[std == 0] = 1.0
    centers_n, labels = kmeans2(features / std, k, iter=10, minit='points', seed=42)
    centers = centers_n * std
    counts  = np.maximum(np.bincount(labels, minlength=k).astype(float), 1.0)
    return centers, counts / counts.sum()


def _build_tree(state, horizon, branches, n_init):
    """Monte Carlo + k-means scenario tree. Returns list of node dicts.

    T_out is deterministic and taken directly from the fixed schedule for every
    node.  Prices and occupancies are uncertain: future values are sampled
    empirically from the historical CSV data at the correct time slot.
    """
    t_now = int(state.get('current_time', 0))
    nodes = [{
        'id': 0, 'stage': 0, 'parent_id': None, 'children': [],
        'price': float(state['price']),
        'price_prev': float(state.get('price_previous', state['price'])),
        'occ1': float(state['occ1']), 'occ2': float(state['occ2']),
        'T_out': _t_out(t_now),   # always from the deterministic schedule
        'prob': 1.0,
    }]
    m_cond   = max(n_init // branches, 30)
    leaf_ids = [0]

    for stage in range(horizon):
        n_smp        = n_init if stage == 0 else m_cond
        new_leaf_ids = []
        # time slot of the children being created at this stage
        t_slot = t_now + stage + 1
        for pid in leaf_ids:
            par = nodes[pid]
            p_s          = _sample_price_csv(t_slot, n_smp)
            o1_s, o2_s   = _sample_occ_csv(t_slot, n_smp)
            centers, prb = _cluster(np.column_stack([p_s, o1_s, o2_s]), branches)
            for b in range(branches):
                cid = len(nodes)
                nodes.append({
                    'id': cid, 'stage': stage + 1, 'parent_id': pid, 'children': [],
                    'price': float(centers[b, 0]), 'price_prev': par['price'],
                    'occ1':  float(centers[b, 1]), 'occ2': float(centers[b, 2]),
                    'T_out': _t_out(t_slot),   # deterministic, same for all branches
                    'prob':  par['prob'] * float(prb[b]),
                })
                par['children'].append(cid)
                new_leaf_ids.append(cid)
        leaf_ids = new_leaf_ids
    return nodes


def _solve_sp(state, nodes):
    """Build and solve the multi-stage stochastic MILP. Returns here-and-now action.

    Dynamics, overrule logic, and ventilation min-up-time mirror the hindsight
    formulation exactly:
      - State variables use wide/free bounds (no hard comfort constraints).
      - Low-temp overrule hysteresis, high-temp cutoff, and humidity overrule are
        modelled with Big-M binary indicators propagated through the scenario tree.
      - Ventilation min-up-time uses the startup-descendant constraint, equivalent
        to the summation form used in the hindsight MILP.
    """
    p        = _P
    T1_obs   = float(state['T1']);   T2_obs  = float(state['T2'])
    H_obs    = float(state['H'])
    occ1_0   = float(state['occ1']); occ2_0  = float(state['occ2'])
    c        = int(state.get('c', 0))
    y_lo1_0  = int(state.get('y_low_1',  0))
    y_lo2_0  = int(state.get('y_low_2',  0))

    # Initial overrule flags from current physical state (same logic as hindsight)
    y_hi1_0  = 1 if T1_obs > p['T_high'] else 0
    y_hi2_0  = 1 if T2_obs > p['T_high'] else 0
    h_ov_0   = 1 if H_obs  > p['H_high'] else 0
    v_prev_0 = 1 if c > 0 else 0

    M_T, M_H  = 60.0, 200.0
    L         = _VENT_UP   # = 3

    ids       = [n['id'] for n in nodes]
    nmap      = {n['id']: n for n in nodes}
    max_stage = max(n['stage'] for n in nodes)

    # ── Descendant lookup for ventilation min-up-time ─────────────────────────
    def _desc(nid, max_d):
        """All descendant ids at depths 1 .. max_d from nid."""
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

    # ── State variables — free bounds, identical to hindsight ─────────────────
    m.T1 = pyo.Var(ids, bounds=(-50.0, 100.0))
    m.T2 = pyo.Var(ids, bounds=(-50.0, 100.0))
    m.H  = pyo.Var(ids, bounds=(-200.0, 500.0))

    # ── Big-M overrule indicator variables (one set per node) ─────────────────
    m.y_lo1  = pyo.Var(ids, domain=pyo.Binary)  # low-temp overrule active, room 1
    m.y_lo2  = pyo.Var(ids, domain=pyo.Binary)  # low-temp overrule active, room 2
    m.z_blo1 = pyo.Var(ids, domain=pyo.Binary)  # T1[nid] ≤ T_low
    m.z_blo2 = pyo.Var(ids, domain=pyo.Binary)  # T2[nid] ≤ T_low
    m.z_bok1 = pyo.Var(ids, domain=pyo.Binary)  # T1[nid] ≤ T_OK
    m.z_bok2 = pyo.Var(ids, domain=pyo.Binary)  # T2[nid] ≤ T_OK
    m.w1     = pyo.Var(ids, domain=pyo.Binary)  # y_lo1 AND z_bok1 (linearised)
    m.w2     = pyo.Var(ids, domain=pyo.Binary)  # y_lo2 AND z_bok2 (linearised)
    m.z_ahi1 = pyo.Var(ids, domain=pyo.Binary)  # T1[nid] > T_high
    m.z_ahi2 = pyo.Var(ids, domain=pyo.Binary)  # T2[nid] > T_high
    m.z_hum  = pyo.Var(ids, domain=pyo.Binary)  # H[nid]  > H_high

    m.cons = pyo.ConstraintList()

    # Fix initial low-temp overrule flags at root
    m.cons.add(m.y_lo1[0] == y_lo1_0)
    m.cons.add(m.y_lo2[0] == y_lo2_0)

    # Root dynamics
    To0 = nmap[0]['T_out']
    m.cons.add(m.T1[0] == T1_obs + p['zeta_exch']*(T2_obs-T1_obs) + p['zeta_loss']*(To0-T1_obs) + p['zeta_conv']*m.p1[0] - p['zeta_cool']*m.v[0] + p['zeta_occ']*occ1_0)
    m.cons.add(m.T2[0] == T2_obs + p['zeta_exch']*(T1_obs-T2_obs) + p['zeta_loss']*(To0-T2_obs) + p['zeta_conv']*m.p2[0] - p['zeta_cool']*m.v[0] + p['zeta_occ']*occ2_0)
    m.cons.add(m.H[0]  == H_obs  + p['eta_occ']*(occ1_0+occ2_0)   - p['eta_vent']*m.v[0])

    # Root overrule constraints from current physical state
    if c > 0:    m.cons.add(m.v[0]  == 1)
    if h_ov_0:   m.cons.add(m.v[0]  == 1)
    if y_hi1_0:  m.cons.add(m.p1[0] == 0.0)
    if y_hi2_0:  m.cons.add(m.p2[0] == 0.0)
    # y_lo1_0/y_lo2_0: enforced via m.y_lo1[0]==y_lo1_0 + low-temp constraint below

    # ── Per-node constraints ───────────────────────────────────────────────────
    for nd in nodes:
        nid = nd['id']
        pid = nd['parent_id']
        s   = nd['stage']

        # (1) Low-temp overrule: y_lo=1 → p forced to P_max
        m.cons.add(m.p1[nid] >= p['P_max'] * m.y_lo1[nid])
        m.cons.add(m.p2[nid] >= p['P_max'] * m.y_lo2[nid])

        # (2) High-temp overrule: parent T > T_high → p = 0 at this node
        if pid is not None:
            m.cons.add(m.p1[nid] <= p['P_max'] * (1 - m.z_ahi1[pid]))
            m.cons.add(m.p2[nid] <= p['P_max'] * (1 - m.z_ahi2[pid]))
        # Root handled above by hard equality (if y_hi1_0/y_hi2_0)

        # (3) Humidity overrule: parent H > H_high → v forced ON at this node
        if pid is not None:
            m.cons.add(m.v[nid] >= m.z_hum[pid])
        # Root handled above (if h_ov_0)

        # (4) Non-root dynamics (same equations as hindsight)
        if pid is not None:
            To = nd['T_out']
            m.cons.add(m.T1[nid] == m.T1[pid] + p['zeta_exch']*(m.T2[pid]-m.T1[pid]) + p['zeta_loss']*(To-m.T1[pid]) + p['zeta_conv']*m.p1[nid] - p['zeta_cool']*m.v[nid] + p['zeta_occ']*nd['occ1'])
            m.cons.add(m.T2[nid] == m.T2[pid] + p['zeta_exch']*(m.T1[pid]-m.T2[pid]) + p['zeta_loss']*(To-m.T2[pid]) + p['zeta_conv']*m.p2[nid] - p['zeta_cool']*m.v[nid] + p['zeta_occ']*nd['occ2'])
            m.cons.add(m.H[nid]  == m.H[pid]  + p['eta_occ']*(nd['occ1']+nd['occ2']) - p['eta_vent']*m.v[nid])

        # (5) Big-M indicator definitions
        # z_blo = 1  iff  T ≤ T_low
        m.cons.add(m.T1[nid] >= p['T_low'] - M_T *  m.z_blo1[nid])
        m.cons.add(m.T1[nid] <= p['T_low'] + M_T * (1 - m.z_blo1[nid]))
        m.cons.add(m.T2[nid] >= p['T_low'] - M_T *  m.z_blo2[nid])
        m.cons.add(m.T2[nid] <= p['T_low'] + M_T * (1 - m.z_blo2[nid]))
        # z_bok = 1  iff  T ≤ T_OK
        m.cons.add(m.T1[nid] >= p['T_OK'] - M_T * (1 - m.z_bok1[nid]))
        m.cons.add(m.T1[nid] <= p['T_OK'] + M_T *    m.z_bok1[nid])
        m.cons.add(m.T2[nid] >= p['T_OK'] - M_T * (1 - m.z_bok2[nid]))
        m.cons.add(m.T2[nid] <= p['T_OK'] + M_T *    m.z_bok2[nid])
        # z_ahi = 1  iff  T > T_high
        m.cons.add(m.T1[nid] <= p['T_high'] + M_T *  m.z_ahi1[nid])
        m.cons.add(m.T1[nid] >= p['T_high'] - M_T * (1 - m.z_ahi1[nid]))
        m.cons.add(m.T2[nid] <= p['T_high'] + M_T *  m.z_ahi2[nid])
        m.cons.add(m.T2[nid] >= p['T_high'] - M_T * (1 - m.z_ahi2[nid]))
        # z_hum = 1  iff  H > H_high  (one-sided Big-M, matching hindsight)
        m.cons.add(m.H[nid] <= p['H_high'] + M_H * m.z_hum[nid])

        # (6) Deactivate low-temp overrule when T ≥ T_OK
        m.cons.add(m.y_lo1[nid] <= m.z_bok1[nid])
        m.cons.add(m.y_lo2[nid] <= m.z_bok2[nid])

        # (7) Propagate low-temp overrule to each child (non-leaf nodes only)
        if nd['children']:
            # Linearise AND: w = y_lo AND z_bok
            m.cons.add(m.w1[nid] <= m.y_lo1[nid])
            m.cons.add(m.w1[nid] <= m.z_bok1[nid])
            m.cons.add(m.w1[nid] >= m.y_lo1[nid] + m.z_bok1[nid] - 1)
            m.cons.add(m.w2[nid] <= m.y_lo2[nid])
            m.cons.add(m.w2[nid] <= m.z_bok2[nid])
            m.cons.add(m.w2[nid] >= m.y_lo2[nid] + m.z_bok2[nid] - 1)
            for cid in nd['children']:
                # y_lo[child] fires if T[parent] ≤ T_low, persists if T[parent] < T_OK
                m.cons.add(m.y_lo1[cid] >= m.z_blo1[nid])
                m.cons.add(m.y_lo1[cid] >= m.w1[nid])
                m.cons.add(m.y_lo1[cid] <= m.z_blo1[nid] + m.w1[nid])
                m.cons.add(m.y_lo2[cid] >= m.z_blo2[nid])
                m.cons.add(m.y_lo2[cid] >= m.w2[nid])
                m.cons.add(m.y_lo2[cid] <= m.z_blo2[nid] + m.w2[nid])

        # (8) Ventilation min-up-time — startup-descendant form
        # If v turns ON at nid (v[nid]=1, v[parent]=0), all descendants within
        # L-1 further levels must also be ON.  Equivalent to the summation
        # constraint used in the hindsight MILP.
        v_prev_nd = v_prev_0 if pid is None else m.v[pid]
        max_d     = min(L - 1, max_stage - s)
        for did in _desc(nid, max_d):
            m.cons.add(m.v[did] >= m.v[nid] - v_prev_nd)

    # ── Force ON for existing inertia at deeper stages ─────────────────────────
    # (equivalent to hindsight's  for tau in range(c0): v[tau]==1)
    for nd in nodes:
        if nd['parent_id'] is not None and c > nd['stage']:
            m.cons.add(m.v[nd['id']] == 1)

    # ── Objective: minimise expected electricity cost ─────────────────────────
    m.obj = pyo.Objective(
        expr=sum(
            nd['prob'] * nd['price'] * (
                m.p1[nd['id']] + m.p2[nd['id']] + p['P_vent'] * m.v[nd['id']]
            )
            for nd in nodes
        ),
        sense=pyo.minimize,
    )

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
# PUBLIC INTERFACE — only function the evaluation framework needs to call
# =============================================================================

def select_action(state: dict) -> dict:
    """
    Multi-stage stochastic programming policy.

    Parameters (hardcoded inside):
        horizon  = 1   look-ahead stages
        branches = 100  k-means clusters per stage (scenario branches)
        n_init   = 10000 Monte Carlo draws before clustering

    Parameters
    ----------
    state : dict
        Current observed state with keys: T1, T2, H, occ1, occ2, price,
        price_previous, c, y_low_1, y_low_2, y_high_1, y_high_2, current_time.

    Returns
    -------
    dict with keys HeatPowerRoom1, HeatPowerRoom2, VentilationON.
    """
    HORIZON  = 2
    BRANCHES = 2
    N_INIT   = 1000

    p = _P

    # At the last time step there is no future to plan for — skip the MILP
    # and fall back directly to the reactive dummy action.
    if int(state.get('current_time', 0)) >= 9:
        return {
            'HeatPowerRoom1': p['P_max'] if state.get('y_low_1') else 0.0,
            'HeatPowerRoom2': p['P_max'] if state.get('y_low_2') else 0.0,
            'VentilationON' : 1 if (int(state.get('c', 0)) > 0
                                    or float(state.get('H', 0)) >= p['H_high']) else 0,
        }

    try:
        nodes = _build_tree(state, HORIZON, BRANCHES, N_INIT)
        return _solve_sp(state, nodes)
    except Exception:
        return {
            'HeatPowerRoom1': p['P_max'] if state.get('y_low_1') else 0.0,
            'HeatPowerRoom2': p['P_max'] if state.get('y_low_2') else 0.0,
            'VentilationON' : 1 if (int(state.get('c', 0)) > 0
                                    or float(state.get('H', 0)) >= p['H_high']) else 0,
        }
