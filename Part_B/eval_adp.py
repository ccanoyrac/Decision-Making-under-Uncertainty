"""Quick evaluation of ADP vs Dummy over the 100-day test dataset.

Run via:  conda run -n 02435_DMUU python eval_adp.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# Use Task6_Environment's run_policy so we evaluate identically to main.ipynb
from Task6_Environment import run_policy, dummy_action
import ADP_policy_14 as adp

price_csv = 'v2_PriceData.csv'
occ1_csv  = 'OccupancyRoom1.csv'
occ2_csv  = 'OccupancyRoom2.csv'

print("Evaluating Dummy policy...")
dummy_costs, dummy_times = run_policy(
    dummy_action, price_csv, occ1_csv, occ2_csv,
    verbose=False)
print(f"  Mean daily cost: {dummy_costs.mean():.3f} euro")
print(f"  Std            : {dummy_costs.std():.3f} euro")
print(f"  Min / Max      : {dummy_costs.min():.3f} / {dummy_costs.max():.3f}")

print("\nEvaluating ADP policy...")
adp_costs, adp_times = run_policy(
    adp.select_action, price_csv, occ1_csv, occ2_csv,
    verbose=False)
print(f"  Mean daily cost: {adp_costs.mean():.3f} euro")
print(f"  Std            : {adp_costs.std():.3f} euro")
print(f"  Min / Max      : {adp_costs.min():.3f} / {adp_costs.max():.3f}")

saving = dummy_costs.mean() - adp_costs.mean()
print(f"\n{'='*50}")
print(f"ADP vs Dummy: {saving:+.3f} euro/day ({'BETTER' if saving>0 else 'WORSE'})")
print(f"Dummy baseline (main.ipynb): 183.046 euro")
print(f"ADP result               : {adp_costs.mean():.3f} euro")
print(f"Beats dummy?             : {'YES' if adp_costs.mean() < dummy_costs.mean() else 'NO'}")
print(f"{'='*50}")