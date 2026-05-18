# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 08:51:59 2025

@author: geots
"""

import numpy as np


def get_fixed_data():
    """
    Returns the fixed data for the heating + ventilation system.
    THIS CODE SHOULD NOT BE CHANGED BY STUDENTS.
    """

    # ------------------------------
    # Simulation settings
    # ------------------------------

    num_timeslots = 10  # number of discrete simulation steps (hours)


    return {

        # Number of timeslots (hours)
        'num_timeslots': num_timeslots,


        # ------------------------------
        # Initial state
        # ------------------------------
        
        "T1": 21.0,  #initial temperature at room 1
        "T2": 21.0, #initial temperature at room 2
        "H": 40.0, #initial humidity
        "Occ1": np.random.uniform(25, 35), #initial occupancy at room 1
        "Occ2": np.random.uniform(15, 25), #initial occupancy at room 2
        "price_t": np.random.uniform(2, 8),  #initial price
        "price_previous": np.random.uniform(2, 8),  #initial previous price
        "vent_counter": 0, # initial counter (the ventilation was not ON previously)
        "low_override_r1": 0,  #initial condition of the overrule controller in room 1 (OFF)
        "low_override_r2": 0, #initial condition of the overrule controller in room 2 (OFF)
        

        # ------------------------------
        # Heating system parameters
        # ------------------------------

        # Maximum heating power (kW)
        # Heater can output between 0 and this value.
        'heating_max_power': 3.0,

        # Heat exchange coefficient between rooms
        # (°C change per hour per °C difference between rooms)
        'heat_exchange_coeff': 0.6,

        # Heating efficiency:
        # Increase in room temperature per kW of heating power (°C per hour per kW)
        'heating_efficiency_coeff': 1.0,

        # Thermal loss coefficient:
        # Fraction of indoor-outdoor temperature difference lost per hour
        # (°C change per hour per °C difference between inddors and outdoors temperature)
        'thermal_loss_coeff': 0.1,

        # Ventilation cooling effect:
        # Temperature decrease in the room for each hour that ventilation is ON (°C)
        'heat_vent_coeff': 0.7,

        # Occupancy heat gain:
        # Temperature increase per hour per person in the room (°C)
        'heat_occupancy_coeff': 0.02,


        # ------------------------------
        # Comfort and control thresholds (°C)
        # ------------------------------

        # Lower threshold for Overrule heater activation
        'temp_min_comfort_threshold': 18.0,

        # Temperature above which the Overrule controller is deactived
        'temp_OK_threshold': 22.0,

        # Hard upper limit: when exceeded, heater must be OFF
        'temp_max_comfort_threshold': 26.0,


        # ------------------------------
        # Outdoor temperature (°C)
        # Known “in hindsight” time series provided to the MILP.
        # A simple sinusoidal profile is used as an example.
        # ------------------------------

        'outdoor_temperature': [
            3 * np.sin(2 * np.pi * t / num_timeslots - np.pi/2)
            for t in range(num_timeslots)
        ],


        # ------------------------------
        # Ventilation system parameters
        # ------------------------------

        # Minimum number of consecutive hours that ventilation must remain ON
        # after being started
        'vent_min_up_time': 3,

        # Humidity threshold above which overrule controller forces ventilation ON (%)
        'humidity_threshold': 70.0,

        # Electrical power consumption of ventilation when ON (kW)
        'ventilation_power': 2.0,
        
        # Degrees of humidity increase per hour per person
        'humidity_occupancy_coeff':0.18,
        
        # Degrees of humidity decrease per hour that ventilation is ON
        'humidity_vent_coeff': 15
        
    }

