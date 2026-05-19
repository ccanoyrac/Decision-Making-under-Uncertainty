# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 12:49:53 2025

@author: geots

Rooms' Occupancy process.
NOT TO BE CHANGED BY THE STUDENTS
"""

import numpy as np


def next_occupancy_levels(r1_current, r2_current):
    """
    Markovian 2-room occupancy update.
    
    Args:
        r1_current (float): Occupancy in Room 1 at time t-1
        r2_current (float): Occupancy in Room 2 at time t-1
    
    Returns:
        (float, float): Occupancies (r1_next, r2_next) at time t
    """

    # ---- Long-run means ----
    mean_r1 = 35.0   # Room 1 is busier
    mean_r2 = 25.0   # Room 2 is less busy

    # ---- Reversion strength (same for both rooms) ----
    rev = 0.25

    # ---- Coupling strength ----
    coupling = 0.1   # weak influence of one room on the other

    # ---- Noise ----
    noise_r1 = np.random.normal(0, 3.0)
    noise_r2 = np.random.normal(0, 2.5)

    # ---- Room 1 update ----
    r1_next = (
        r1_current
        + rev * (mean_r1 - r1_current)
        + coupling * (r2_current - r1_current)
        + noise_r1
    )

    # ---- Room 2 update ----
    r2_next = (
        r2_current
        + rev * (mean_r2 - r2_current)
        + coupling * (r1_current - r2_current)
        + noise_r2
    )

    # ---- Enforce boundaries ----
    r1_next = float(np.clip(r1_next, 20, 50))
    r2_next = float(np.clip(r2_next, 10, 30))

    return r1_next, r2_next



### Example use to generate trajectories


import matplotlib.pyplot as plt

def generate_trajectories(T, num_paths):

    r1_paths = []
    r2_paths = []

    for _ in range(num_paths):
        # Random initial states inside allowed ranges
        r1 = [np.random.uniform(25, 35)]
        r2 = [np.random.uniform(15, 25)]

        for t in range(1, T):
            r1_next, r2_next = next_occupancy_levels(r1[-1], r2[-1])
            r1.append(r1_next)
            r2.append(r2_next)

        r1_paths.append(r1)
        r2_paths.append(r2)

    return r1_paths, r2_paths


def plot_trajectories(r1_paths, r2_paths):
    T = len(r1_paths[0])

    # --- Room 1 ---
    plt.figure(figsize=(8,4))
    for traj in r1_paths:
        plt.plot(traj)
    plt.title("Room 1 Occupancy Trajectories (20–50)")
    plt.xlabel("Time")
    plt.ylabel("Occupancy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Room 2 ---
    plt.figure(figsize=(8,4))
    for traj in r2_paths:
        plt.plot(traj)
    plt.title("Room 2 Occupancy Trajectories (10–30)")
    plt.xlabel("Time")
    plt.ylabel("Occupancy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    r1, r2 = generate_trajectories(T=10, num_paths=100)
    plot_trajectories(r1, r2)

#import pandas as pd

#pd.DataFrame(r1).to_csv("OutOfSampleOccupancyRoom1.csv", index=False)
#pd.DataFrame(r2).to_csv("OutOfSampleOccupancyRoom2.csv", index=False)


