import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import ADP_policy_14 as adp
import v2_SystemCharacteristics as sc
import numpy as np

raw = sc.get_fixed_data()
p = {
    'zeta_exch': raw['heat_exchange_coeff'],
    'zeta_loss': raw['thermal_loss_coeff'],
    'zeta_conv': raw['heating_efficiency_coeff'],
    'zeta_cool': raw['heat_vent_coeff'],
    'zeta_occ' : raw['heat_occupancy_coeff'],
    'eta_occ'  : raw['humidity_occupancy_coeff'],
    'eta_vent' : raw['humidity_vent_coeff'],
    'T_out'    : raw['outdoor_temperature'],
    'P_max'    : raw['heating_max_power'],
    'P_vent'   : raw['ventilation_power'],
    'T_low'    : raw['temp_min_comfort_threshold'],
    'T_ok'     : raw['temp_OK_threshold'],
    'H_high'   : raw['humidity_threshold'],
}

def apply_one_step(T1, T2, H, occ1, occ2, p1, p2, v, t):
    T_out = float(p['T_out'][min(t,9)])
    T1n = T1 + p['zeta_exch']*(T2-T1) + p['zeta_loss']*(T_out-T1) + p['zeta_conv']*p1 - p['zeta_cool']*v + p['zeta_occ']*occ1
    T2n = T2 + p['zeta_exch']*(T1-T2) + p['zeta_loss']*(T_out-T2) + p['zeta_conv']*p2 - p['zeta_cool']*v + p['zeta_occ']*occ2
    Hn  = max(0, H + p['eta_occ']*(occ1+occ2) - p['eta_vent']*v)
    return T1n, T2n, Hn

def dummy(state):
    c, H = state.get('c',0), state.get('H',0)
    v = 1 if (c > 0 or H >= p['H_high']) else 0
    p1 = p['P_max'] if state.get('y_low_1') else 0.0
    p2 = p['P_max'] if state.get('y_low_2') else 0.0
    return {'HeatPowerRoom1': p1, 'HeatPowerRoom2': p2, 'VentilationON': v}

def rollout(policy_fn, prices, label):
    T1, T2, H = 21.0, 21.0, 40.0
    occ1, occ2 = 30.0, 20.0
    prev_p = prices[0]
    total = 0.0
    y_lo1, y_lo2 = 0, 0
    print(f"\n=== {label} (prices: {[f'{x:.1f}' for x in prices]}) ===")
    for t in range(10):
        state = {
            'T1': T1, 'T2': T2, 'H': H, 'c': 0,
            'y_low_1': y_lo1, 'y_low_2': y_lo2, 'y_high_1': 0, 'y_high_2': 0,
            'occ1': occ1, 'occ2': occ2,
            'price': prices[t], 'price_previous': prev_p,
            'current_time': t,
        }
        a = policy_fn(state)
        # apply overrule
        p1 = float(a['HeatPowerRoom1'])
        p2 = float(a['HeatPowerRoom2'])
        v  = int(a['VentilationON'])
        if y_lo1: p1 = p['P_max']
        if y_lo2: p2 = p['P_max']
        cost = prices[t] * (p1 + p2 + p['P_vent']*v)
        total += cost
        T1n, T2n, Hn = apply_one_step(T1, T2, H, occ1, occ2, p1, p2, v, t)
        y_lo1 = 1 if T1n < p['T_low'] else (0 if T1n >= p['T_ok'] else y_lo1)
        y_lo2 = 1 if T2n < p['T_low'] else (0 if T2n >= p['T_ok'] else y_lo2)
        print(f"  t={t} | T1={T1:.1f}→{T1n:.2f} p1={p1:.2f} | cost={cost:.2f}€ | y_lo={y_lo1}")
        prev_p = prices[t]
        T1, T2, H = T1n, T2n, Hn
    print(f"  TOTAL: {total:.2f}€")
    return total

# Low-price day
lp = [2.0, 2.1, 1.9, 2.2, 2.3, 2.0, 1.8, 2.1, 2.4, 2.2]
print("\n" + "="*60)
adp_lp = rollout(adp.select_action, lp, "ADP — LOW price day")
dummy_lp = rollout(dummy, lp, "Dummy — LOW price day")

# Medium-price day
mp = [4.5, 4.8, 5.2, 5.0, 4.3, 3.9, 3.5, 3.8, 4.1, 4.0]
print("\n" + "="*60)
adp_mp = rollout(adp.select_action, mp, "ADP — MEDIUM price day")
dummy_mp = rollout(dummy, mp, "Dummy — MEDIUM price day")

print(f"\n{'='*60}")
print(f"LOW-price:    ADP={adp_lp:.2f}€  Dummy={dummy_lp:.2f}€  ADP saves {dummy_lp-adp_lp:.2f}€")
print(f"MEDIUM-price: ADP={adp_mp:.2f}€  Dummy={dummy_mp:.2f}€  ADP saves {dummy_mp-adp_mp:.2f}€")