# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 13:14:45 2025

@author: geots
"""
import matplotlib.pyplot as plt
import numpy as np

# Fixed plot_HVAC_results function - adapted from PlotsRestaurant.py
def plot_HVAC_results_fixed(T,Temp_r1, Temp_r2, h_r1, h_r2, v, Hum, price, Occ_r1, Occ_r2):
    """
    Plot HVAC system results with temperatures, heating, ventilation, humidity, and occupancy.
    Handles different array lengths from different cell preparations.
    """

    
    # Determine correct array lengths and pad/match as needed
    n_decision = len(h_r1)  # Use heating as reference (10 elements - decision variables)
    n_state = len(Hum)      # State variables including initial time (10 or 11 elements)
    
    # Pad v to match decision variables if needed
    v_padded = v.copy()
    if len(v_padded) < n_decision:
        v_padded = np.pad(v_padded, (0, n_decision - len(v_padded)), mode='constant', constant_values=0)
    elif len(v_padded) > n_decision:
        v_padded = v_padded[:n_decision]
    
    # Adjust price and occupancy if needed (should be 10 elements)
    price_padded = price[:n_decision] if len(price) >= n_decision else price
    occ_r1_padded = Occ_r1[:n_decision] if len(Occ_r1) >= n_decision else Occ_r1
    occ_r2_padded = Occ_r2[:n_decision] if len(Occ_r2) >= n_decision else Occ_r2
    
    # Adjust temperatures for state variables
    temp_r1_plot = Temp_r1[:n_state] if len(Temp_r1) >= n_state else Temp_r1
    temp_r2_plot = Temp_r2[:n_state] if len(Temp_r2) >= n_state else Temp_r2
    
    # Time indices
    time_decisions = np.arange(n_decision)  # For decision variables
    time_states = np.arange(n_state)       # For state variables
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(14, 14))

    # ===== SUBPLOT 1: Room Temperatures =====
    axes[0].plot(time_states, temp_r1_plot, label='Room 1 Temp', marker='o', linewidth=2, markersize=5)
    axes[0].plot(time_states, temp_r2_plot, label='Room 2 Temp', marker='s', linewidth=2, markersize=5)
    axes[0].axhline(18, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Min Comfort (18°C)')
    axes[0].axhline(22, color='green', linestyle='--', alpha=0.7, linewidth=2, label='OK (22°C)')
    axes[0].axhline(26, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Max (26°C)')
    axes[0].fill_between(time_states, 18, 22, alpha=0.1, color='green')
    axes[0].set_ylabel("Temperature (°C)", fontweight='bold', fontsize=11)
    axes[0].set_title("Room Temperatures", fontweight='bold', fontsize=12)
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # ===== SUBPLOT 2: Heater Consumption =====
    axes[1].bar(time_decisions - 0.2, h_r1[:n_decision], width=0.4, label='Room 1 Heater', alpha=0.7, color='tab:blue')
    axes[1].bar(time_decisions + 0.2, h_r2[:n_decision], width=0.4, label='Room 2 Heater', alpha=0.7, color='tab:orange')
    axes[1].plot(time_decisions, h_r1[:n_decision] + h_r2[:n_decision], color='red', marker='D', 
                 label='Total Heating', linewidth=2, markersize=4)
    axes[1].set_ylabel("Heater Power (kW)", fontweight='bold', fontsize=11)
    axes[1].set_title("Heater Consumption", fontweight='bold', fontsize=12)
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    # ===== SUBPLOT 3: Ventilation and Humidity =====
    axes[2].step(time_decisions, v_padded, where='mid', label='Ventilation ON', color='tab:blue', linewidth=2.5)
    axes[2].fill_between(time_decisions, 0, v_padded, step='mid', alpha=0.2, color='tab:blue')
    ax2_twin = axes[2].twinx()
    ax2_twin.plot(time_states, Hum, label='Humidity (%)', color='tab:green', marker='o', linewidth=2, markersize=5)
    ax2_twin.axhline(70, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Threshold (70%)')
    axes[2].set_ylabel("Ventilation Status", fontweight='bold', fontsize=11)
    ax2_twin.set_ylabel("Humidity (%)", fontweight='bold', fontsize=11, color='tab:green')
    axes[2].set_title("Ventilation Status and Humidity", fontweight='bold', fontsize=12)
    axes[2].set_ylim([-0.1, 1.1])
    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    axes[2].grid(True, alpha=0.3)
    
    # ===== SUBPLOT 4: Electricity Price and Occupancy =====
    axes[3].plot(time_decisions, price_padded, label='TOU Price (€/kWh)', color='red', marker='x', 
                 linewidth=2.5, markersize=7)
    axes[3].set_ylabel("Price (€/kWh)", fontweight='bold', fontsize=11, color='red')
    axes[3].tick_params(axis='y', labelcolor='red')
    
    ax3_twin = axes[3].twinx()
    ax3_twin.bar(time_decisions - 0.2, occ_r1_padded, width=0.4, label='Occupancy Room 1', 
                 alpha=0.6, color='steelblue')
    ax3_twin.bar(time_decisions + 0.2, occ_r2_padded, width=0.4, label='Occupancy Room 2', 
                 alpha=0.6, color='coral')
    ax3_twin.set_ylabel("Occupancy (people)", fontweight='bold', fontsize=11)
    axes[3].set_xlabel("Time (hours)", fontweight='bold', fontsize=11)
    axes[3].set_title("Electricity Price and Occupancy", fontweight='bold', fontsize=12)
    
    lines1, labels1 = axes[3].get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    axes[3].legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
