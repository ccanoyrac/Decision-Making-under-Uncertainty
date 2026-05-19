# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 12:49:53 2025

@author: geots
"""

"""
Price process.
NOT TO BE CHANGED BY THE STUDENTS
"""

import numpy as np
import matplotlib.pyplot as plt
from Data import v2_SystemCharacteristics as v2_SystemCharacteristics


def price_model(current_price, previous_price):
    """
    Price process with dependence on previous prices.
    """
    mean_price = 4
    reversion_strength = 0.12
    price_cap = 12
    price_floor = 0

    mean_reversion = reversion_strength * (mean_price - current_price)
    noise = np.random.normal(0, 0.5)

    next_price = current_price + 0.6 * (current_price - previous_price) + mean_reversion + noise

    # Special handling if price goes negative
    if next_price < 0:
        if np.random.rand() > 0.2:
            next_price = np.random.uniform(0, mean_price * 0.3)

    # Enforce bounds
    return max(min(next_price, price_cap), price_floor)


# -----------------------------
# Example Use: Generate and plot trajectories
# -----------------------------
params = v2_SystemCharacteristics.get_fixed_data()
T = int(params['num_timeslots'])

num_paths = 100
all_paths = []

for i in range(num_paths):
    # initialize with current price
    traj = [np.random.uniform(2, 8)]

    for t in range(1, T):
        prev = traj[-1]
        prev_prev = traj[-2] if t > 1 else 6 # arbitrary previous price for t=1
        traj.append(price_model(prev, prev_prev))

    all_paths.append(traj)

"""
# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 5))

for i, traj in enumerate(all_paths):
    plt.plot(range(T), traj, alpha=0.8, label=f"Path {i+1}")

plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Sample Electricity Price Trajectories")
plt.grid(True, linestyle="--", alpha=0.4)
#plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()

import pandas as pd

#pd.DataFrame(all_paths).to_csv("OutOfSamplePriceData.csv", index=False)

"""