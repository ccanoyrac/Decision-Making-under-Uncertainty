# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 08:51:59 2025

@author: geots
"""

import numpy as np
import matplotlib.pyplot as plt 
from sympy import pprint
import pprint
import pandas as pd
import os
import json

def get_fixed_data():
    """
    Returns the fixed data for the heating + ventilation system.
    THIS CODE SHOULD NOT BE CHANGED BY STUDENTS.
    """

    # ------------------------------
    # Simulation settings
    # ------------------------------
    Temperature_outside_base = 10.0  # base outdoor temperature (°C)
    num_timeslots = 10  # number of discrete simulation steps (hours)
    initial_temperature = 21.0  # initial indoor temperature (°C)
    previous_initial_temperature = 21.0  # initial previous indoor temperature (°C)
    initial_humidity = 40.0  # initial indoor humidity (%)
    heating_max_power = 3.0  # maximum heating power (kW)
    heat_exchange_coeff = 0.6  # heat exchange coefficient between rooms (°C change per hour per °C difference between rooms)
    heating_efficiency_coeff = 1.0  # heating efficiency (°C per hour per kW)
    thermal_loss_coeff = 0.1  # thermal loss coefficient (°C change per hour per °C difference between indoors and outdoors temperature)
    heat_vent_coeff = 0.7  # ventilation cooling effect (°C decrease in the room for each hour that ventilation is ON)
    heat_occupancy_coeff = 0.02  # occupancy heat gain (°C increase per hour per person in the room)
    temp_min_comfort_threshold = 18.0  # lower threshold for Overrule heater activation (°C)
    temp_OK_threshold = 22.0  # temperature above which the Overrule controller is deactived (°C)
    temp_max_comfort_threshold = 26.0  # hard upper limit: when exceeded, heater must be OFF (°C)
    vent_min_up_time = 3  # minimum number of consecutive hours that ventilation must remain ON after being started
    humidity_threshold = 70.0  # humidity threshold above which overrule controller forces ventilation ON (%)
    ventilation_power = 2.0  # electrical power consumption of ventilation when ON (kW)
    humidity_occupancy_coeff = 0.18  # degrees of humidity increase per hour per person
    humidity_vent_coeff = 15  # degrees of humidity decrease per hour that ventilation is ON    
    prices = get_price_data()
    occupacy_room_1 = get_occupancy_room1_data()
    occupacy_room_2 = get_occupancy_room2_data()
    
    return {
        # Number of timeslots (hours)
        'num_timeslots': num_timeslots,


        # ------------------------------
        # Initial indoor conditions
        # ------------------------------
        'initial_temperature': initial_temperature,
        'previous_initial_temperature': previous_initial_temperature,
        'initial_humidity': initial_humidity,


        # ------------------------------
        # Heating system parameters
        # ------------------------------

        # Maximum heating power (kW)
        # Heater can output between 0 and this value.
        'heating_max_power': heating_max_power,

        # Heat exchange coefficient between rooms
        # (°C change per hour per °C difference between rooms)
        'heat_exchange_coeff': heat_exchange_coeff,

        # Heating efficiency:
        # Increase in room temperature per kW of heating power (°C per hour per kW)
        'heating_efficiency_coeff': heating_efficiency_coeff,

        # Thermal loss coefficient:
        # Fraction of indoor-outdoor temperature difference lost per hour
        # (°C change per hour per °C difference between inddors and outdoors temperature)
        'thermal_loss_coeff': thermal_loss_coeff,

        # Ventilation cooling effect:
        # Temperature decrease in the room for each hour that ventilation is ON (°C)
        'heat_vent_coeff': heat_vent_coeff,

        # Occupancy heat gain:
        # Temperature increase per hour per person in the room (°C)
        'heat_occupancy_coeff': heat_occupancy_coeff,


        # ------------------------------
        # Comfort and control thresholds (°C)
        # ------------------------------

        # Lower threshold for Overrule heater activation
        'temp_min_comfort_threshold':  temp_min_comfort_threshold,

        # Temperature above which the Overrule controller is deactived
        'temp_OK_threshold': temp_OK_threshold,

        # Hard upper limit: when exceeded, heater must be OFF
        'temp_max_comfort_threshold': temp_max_comfort_threshold,


        # ------------------------------
        # Outdoor temperature (°C)
        # Known “in hindsight” time series provided to the MILP.
        # A simple sinusoidal profile is used as an example.
        # ------------------------------

        'outdoor_temperature': [
            Temperature_outside_base + 3 * np.sin(2 * np.pi * t / num_timeslots - np.pi/2)
            for t in range(num_timeslots)
        ],


        # ------------------------------
        # Ventilation system parameters
        # ------------------------------

        # Minimum number of consecutive hours that ventilation must remain ON
        # after being started
        'vent_min_up_time': vent_min_up_time,

        # Humidity threshold above which overrule controller forces ventilation ON (%)
        'humidity_threshold': humidity_threshold,

        # Electrical power consumption of ventilation when ON (kW)
        'ventilation_power': ventilation_power,
        
        # Degrees of humidity increase per hour per person
        'humidity_occupancy_coeff': humidity_occupancy_coeff,
        
        # Degrees of humidity decrease per hour that ventilation is ON
        'humidity_vent_coeff': humidity_vent_coeff,
        
        # ------------------------------
        # Price data
        'prices': prices,

        # ------------------------------
        # Occupancy data for room 1 and room 2  
        'occupacy_room_1': occupacy_room_1,
        'occupacy_room_2': occupacy_room_2

        
    }

def get_price_data():
    """
    Reads the PriceData.csv file and returns a dictionary with prices for each time slot.
    Each row represents a day and each column represents an hour.
    
    Returns:
        dict: A dictionary with the following structure:
              - 'flat': A flat dictionary with (day, hour) tuples as keys
                        Example: {(0, 0): 3.488, (0, 1): 2.237, ..., (89, 9): 4.862}
              - 'nested': A nested dictionary with days, then hours
                          Example: {0: {0: 3.488, 1: 2.237, ...}, 1: {...}, ...}
              - 'raw': The raw data as a 2D numpy array
    """
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'PriceData.csv')
    
    # Read the CSV file
    price_df = pd.read_csv(csv_path, header=None)
    
    # Create flat dictionary with (day, hour) tuples as keys
    flat_prices = {}
    for day in range(len(price_df)):
        for hour in range(len(price_df.columns)):
            flat_prices[(day, hour)] = price_df.iloc[day, hour]
    
    # Create nested dictionary with days as keys and hour:price dictionaries as values
    nested_prices = {}
    for day in range(len(price_df)):
        nested_prices[day] = {}
        for hour in range(len(price_df.columns)):
            nested_prices[day][hour] = price_df.iloc[day, hour]
    
    return {
        #'flat': flat_prices,
        'nested': nested_prices,
        #'raw': price_df.values
    }

def get_occupancy_room1_data():
    """
    Reads the OccupancyRoom1.csv file and returns a dictionary with occupancy for each time slot.
    Each row represents a day and each column represents an hour.
    
    Returns:
        dict: A dictionary with the following structure:
              - 'flat': A flat dictionary with (day, hour) tuples as keys
                        Example: {(0, 0): 15.47, (0, 1): 19.41, ..., (89, 9): 26.05}
              - 'nested': A nested dictionary with days, then hours
                          Example: {0: {0: 15.47, 1: 19.41, ...}, 1: {...}, ...}
              - 'raw': The raw data as a 2D numpy array
    """
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'OccupancyRoom1.csv')
    
    # Read the CSV file
    occupancy_df = pd.read_csv(csv_path, header=None)
    
    # Create flat dictionary with (day, hour) tuples as keys
    flat_occupancy = {}
    for day in range(len(occupancy_df)):
        for hour in range(len(occupancy_df.columns)):
            flat_occupancy[(day, hour)] = occupancy_df.iloc[day, hour]
    
    # Create nested dictionary with days as keys and hour:occupancy dictionaries as values
    nested_occupancy = {}
    for day in range(len(occupancy_df)):
        nested_occupancy[day] = {}
        for hour in range(len(occupancy_df.columns)):
            nested_occupancy[day][hour] = occupancy_df.iloc[day, hour]
    
    return {
        #'flat': flat_occupancy,
        'nested': nested_occupancy,
        #'raw': occupancy_df.values
    }

def get_occupancy_room2_data():
    """
    Reads the OccupancyRoom2.csv file and returns a dictionary with occupancy for each time slot.
    Each row represents a day and each column represents an hour.
    
    Returns:
        dict: A dictionary with the following structure:
              - 'flat': A flat dictionary with (day, hour) tuples as keys
                        Example: {(0, 0): 15.47, (0, 1): 19.41, ..., (89, 9): 26.05}
              - 'nested': A nested dictionary with days, then hours
                          Example: {0: {0: 15.47, 1: 19.41, ...}, 1: {...}, ...}
              - 'raw': The raw data as a 2D numpy array
    """
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'OccupancyRoom2.csv')
    
    # Read the CSV file
    occupancy_df = pd.read_csv(csv_path, header=None)
    
    # Create flat dictionary with (day, hour) tuples as keys
    flat_occupancy = {}
    for day in range(len(occupancy_df)):
        for hour in range(len(occupancy_df.columns)):
            flat_occupancy[(day, hour)] = occupancy_df.iloc[day, hour]
    
    # Create nested dictionary with days as keys and hour:occupancy dictionaries as values
    nested_occupancy = {}
    for day in range(len(occupancy_df)):
        nested_occupancy[day] = {}
        for hour in range(len(occupancy_df.columns)):
            nested_occupancy[day][hour] = occupancy_df.iloc[day, hour]
    
    return {
        #'flat': flat_occupancy,
        'nested': nested_occupancy,
        #'raw': occupancy_df.values
    }


current_dir = os.path.dirname(os.path.abspath(__file__))
json_dir = os.path.join(current_dir, 'Data_JSON')

# Create Data_JSON folder if it doesn't exist
if not os.path.exists(json_dir):
    os.makedirs(json_dir)

# Get data from all functions
price_data = get_price_data()
occupancy_room1_data = get_occupancy_room1_data()
occupancy_room2_data = get_occupancy_room2_data()
fixed_data = get_fixed_data()

# Convert to JSON-serializable format (convert keys to strings for flat dicts)
price_json = {'nested': price_data['nested']}
occupancy_room1_json = {'nested': occupancy_room1_data['nested']}
occupancy_room2_json = {'nested': occupancy_room2_data['nested']}

# Convert numpy arrays to lists for JSON serialization
fixed_data_json = fixed_data.copy()
if isinstance(fixed_data_json['outdoor_temperature'], list):
    fixed_data_json['outdoor_temperature'] = [float(x) for x in fixed_data_json['outdoor_temperature']]

# Write to JSON files in Data_JSON folder

with open(os.path.join(json_dir, 'FixedData.json'), 'w') as f:
    json.dump(fixed_data_json, f, indent=2)

print("JSON files created successfully in Data_JSON folder!")
#pprint.pprint(get_fixed_data())





