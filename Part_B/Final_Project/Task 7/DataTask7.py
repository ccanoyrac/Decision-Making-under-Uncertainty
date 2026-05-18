# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 12:30:18 2025

@author: geots
"""

import numpy as np


def fetch_data():
    """
    Returns the fixed data for Task 7.
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
        # Mall power capacity
        # ------------------------------
        
        'P_mall': 45,
        
        # ------------------------------
        # Reference Temperature (°C) - the same for all stores and hours
        # ------------------------------

        'Temperature_reference': 21,


        # ------------------------------
        # Initial indoor conditions
        # ------------------------------
        'initial_temperature': 21.0,
        

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
        # Outdoor temperature (°C)
        # Known “in hindsight” time series provided.
        # A simple sinusoidal profile is used as an example.
        # ------------------------------

        'outdoor_temperature': [
            3 * np.sin(2 * np.pi * t / num_timeslots - np.pi/2)
            for t in range(num_timeslots)
        ],


        
        
        
        
    }
