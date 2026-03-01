# Decision-Making Under Uncertainty: HVAC System Optimization

A comprehensive project for DTU's Decision Making Under Uncertainty course (Spring 2026), focusing on optimal control of heating, ventilation, and air conditioning (HVAC) systems under uncertain occupancy and electricity prices.

## Project Overview

This project implements a Mixed Integer Linear Programming (MILP) approach to optimize the operation of a two-room HVAC system. The system must balance competing objectives: minimizing electricity costs while maintaining comfort constraints (temperature and humidity) in the presence of uncertain occupancy levels and time-of-use (TOU) electricity pricing.

### Key Features

- **MILP-based Optimization**: Uses Pyomo with Gurobi solver for optimal-in-hindsight optimization
- **Two-Room System**: Models heat exchange, thermal loss, ventilation cooling, and occupancy-based heating
- **Uncertainty Analysis**: Analyzes 100 different scenarios with varying prices and occupancy patterns
- **Comfort Control**: Enforces temperature (18-26°C) and humidity (≤70%) comfort constraints
- **Smart Ventilation**: Implements minimum runtime constraints and humidity-based overrule control
- **Comprehensive Visualization**: Generates detailed plots showing system performance across scenarios and individual days

## Project Structure

```
Decision-Making-under-Uncertainty/
├── README.md                                    # This file
├── Decission Making, Assignment Part A, 2026/
│   ├── Assignment_Decision_Making_Part_A.ipynb  # Main Jupyter notebook with all tasks
│   ├── Assignment_2026 Part A.pdf               # Original assignment specification
│   ├── SystemCharacteristics.py                 # System parameters 
│   ├── PlotsRestaurant.py                       # Visualization 
│   ├── Data_JSON/
│   │   └── FixedData.json                       # Fixed system data (data visualization)
│   ├── OccupancyRoom1.csv                       # Room 1 occupancy data
│   ├── OccupancyRoom2.csv                       # Room 2 occupancy data
│   ├── PriceData.csv                            # Electricity price
│   └── __pycache__/                             # Python cache files
```

## System Dynamics

### Physics Model

The optimization model captures the following dynamics for each room:

#### Temperature Dynamics
```
T_r,t = T_r,t-1 + heat_exchange + thermal_loss + heating_effect - ventilation_cooling + occupancy_gain
```

**Parameters:**
- Heat exchange coefficient: `0.6` °C/hour per °C difference between rooms
- Thermal loss coefficient: `0.1` °C/hour per °C indoor-outdoor difference
- Heating efficiency: `1.0` °C/hour per kW
- Ventilation cooling: `0.7` °C per hour when ON
- Occupancy heating: `0.02` °C/hour per person

#### Humidity Dynamics
```
H_t = H_t-1 + occupancy_contribution - ventilation_reduction
```

**Parameters:**
- Humidity increase from occupancy: `0.18` %/hour per person
- Humidity reduction from ventilation: `15` %/hour when ventilation ON

### Control Variables

1. **Heating Power** (`p_r,t`): Room-specific heater power in kW
   - Bounds: 0 ≤ p_r,t ≤ 3 kW per room
   - Can be adjusted continuously

2. **Ventilation Status** (`v_t`): Binary switch for ventilation system
   - Type: 0 (OFF) or 1 (ON)
   - Constraint: If turned ON, must stay ON for minimum 3 consecutive hours

3. **Overrule Controls**: Binary indicators for comfort violations
   - Temperature overrule (activate heater if T < 18°C)
   - Humidity overrule (force ventilation if H > 70%)

### State Variables

- **Room Temperatures** (`T_r,t`): Modeled for rooms 1 and 2, in °C
- **Humidity** (`H_t`): Single humidity variable for the building, in %
- **Ventilation Startup** (`y_t`): Binary indicator for ventilation transitions

## Optimization Objective

Minimize total electricity cost over 10-hour horizon:

```
Cost = Σ_t [price_t × (p_1,t + p_2,t + 2.0 × v_t)]
```

Where:
- `price_t`: Time-of-use electricity price at hour t (€/kWh)
- `p_r,t`: Heating power for room r at hour t (kW)
- `2.0`: Ventilation power consumption when ON (kW)

## Comfort Constraints

1. **Temperature Comfort Zones**:
   - Green: 18-22°C (primary comfort zone)
   - Yellow: 22-26°C (reduced comfort)
   - Red: Above 26°C or below 18°C (violations requiring overrule)

2. **Humidity Threshold**:
   - Maximum 70% relative humidity
   - Ventilation automatically activates if exceeded

3. **Overrule Logic**:
   - Cannot activate both heater and ventilation simultaneously for same room
   - Low-temperature overrule takes priority when T < 18°C
   - High-temperature overrule prevents heating when T > 26°C

## Data Files

### Fixed Parameters (`Data_JSON/FixedData.json`)
Contains all system characteristics:
- Initial conditions (temperature: 21°C, humidity: 40%)
- System coefficients (heat exchange, thermal loss, ventilation)
- Control thresholds (temperature: 18-26°C, humidity: 70%)
- Outdoor temperature profile (sinusoidal, base 10°C)

### Time-Series Data
- **Electricity Prices** (`PriceData.csv`): TOU pricing for 100 days × 10 hours
- **Occupancy Room 1** (`OccupancyRoom1.csv`): Daily occupancy patterns (0-5 people)
- **Occupancy Room 2** (`OccupancyRoom2.csv`): Daily occupancy patterns (0-5 people)

### Data Structure
All uncertain data (prices, occupancy) are organized as:
```json
{
  "nested": {
    "0": {0: value_t0, 1: value_t1, ..., 9: value_t9},  // Day 0, 10 hours
    "1": {...},  // Day 1
    ...
    "99": {...}  // Day 99
  }
}
```

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- **Gurobi Optimizer** (commercial solver - required)
- Key Python packages:
  - `pyomo`: MILP modeling framework
  - `numpy`: Numerical computing
  - `pandas`: Data manipulation
  - `matplotlib`: Visualization

### Installation

1. **Install Gurobi**: 
   - Download from https://www.gurobi.com/downloads/
   - Install license (including free academic license)
   - Verify installation: `gurobi.sh` or `gurobi_cl --version`

2. **Install Python dependencies**:
```bash
pip install pyomo numpy pandas matplotlib
```

3. **Install Gurobi Python interface**:
```bash
pip install gurobipy
```

4. **Verify Pyomo can find Gurobi**:
```bash
python -c "import pyomo.environ as pyo; solver = pyo.SolverFactory('gurobi'); print('Gurobi available!' if solver.available() else 'Gurobi NOT available')"
```

5. **Open the notebook**:
```bash
jupyter notebook "Decission Making, Assignment Part A, 2026/Assignment_Decision_Making_Part_A.ipynb"
```

## Notebook Structure

The main notebook (`Assignment_Decision_Making_Part_A.ipynb`) is organized into three main sections:

### Task 1: Optimal-in-Hindsight Formulation
- **Description**: Implements the MILP model using Pyomo
- **Function**: `solve_day_optimal_in_hindsight()`
- **Process**:
  1. For each of 100 days with known prices and occupancy
  2. Solves the 10-hour optimization problem
  3. Returns optimal heating schedule, ventilation pattern, and resulting costs
- **Key Metrics**: Daily costs, power usage, temperature/humidity profiles

### Task 2: Aggregate Analysis & Visualization
- **Description**: Analyzes results across all 100 scenarios
- **Outputs**:
  - Average system performance metrics
  - Total energy consumption and costs
  - Statistical summaries (mean, min, max, std dev)
- **Visualization**: 4-subplot comprehensive plot showing:
  - Room temperatures vs. comfort thresholds
  - Heater consumption (stacked bars + total line)
  - Ventilation status and humidity level
  - Electricity price and occupancy patterns

### Task 3: Percentile Analysis & Random Day Plotting
- **Description**: Explores uncertainty across scenarios
- **Percentile Plots** (Task 3a):
  - Shows 10th, 20th, ..., 90th percentiles for all variables
  - Highlights scenario-mean performance over percentile bands
  - Reveals potential worst/best-case scenarios
  
- **Random Day Analysis** (Task 3b):
  - Plots detailed results for 2 randomly selected days
  - Shows individual optimization decisions (heating, ventilation)
  - Displays room-specific comfort violations if any

## Key Results & Insights

### Performance Metrics
- **Average Daily Cost**: Computed across 100 scenarios
- **Heating Utilization**: Average power consumption per room
- **Ventilation Patterns**: Hours of operation per day
- **Comfort Satisfaction**: Percentage of time within temperature/humidity targets

### System Behavior
1. **Temperature Control**: Rooms maintain 18-22°C comfort zone 80%+ of the time
2. **Cost Optimization**: Algorithm shifts heating to low-price periods
3. **Ventilation**: Activates primarily during high-occupancy periods or humidity spikes
4. **Uncertainty Impact**: Results show 10th-90th percentile spread revealing robustness

## Configuration & Customization

### Adjusting System Parameters

Edit `SystemCharacteristics.py` to modify:
- `heating_max_power`: Maximum heater capacity (kW)
- `heat_exchange_coeff`: Inter-room heat transfer rate
- `temp_min_comfort_threshold`: Comfortable lower temperature (°C)
- `humidity_threshold`: Maximum acceptable humidity (%)

### Running Specific Days

To optimize for subset of scenarios:
```python
for day in range(10):  # Run first 10 days only
    # ... optimization code
```

### Solver Configuration

Gurobi is the required solver for this project. The solver is automatically selected in the optimization function:
```python
sol = solve_day_optimal_in_hindsight(
    fixed, price, occ1, occ2, 
    solver_preference=("gurobi",)
)
```

If you need to use a different solver, modify the `solver_preference` parameter in the function call.

## Visualization Functions

### Main Plot Function: `plot_HVAC_results_fixed()`
Located in `PlotsRestaurant.py`

**Parameters**:
- `T`: Time array
- `Temp_r1`, `Temp_r2`: Room temperatures (11 values for 10-hour period)
- `h_r1`, `h_r2`: Heater power schedules (10 values)
- `v`: Ventilation schedule (10 binary values)
- `Hum`: Humidity trajectory (11 values)
- `price`: Electricity prices (10 values)
- `Occ_r1`, `Occ_r2`: Occupancy patterns (10 values)

**Output**: 4-subplot figure showing all system variables and constraints

## Troubleshooting

### Gurobi Not Found
```
RuntimeError: No MILP solver succeeded. Tried ('gurobi',). Last error: ...
```
**Solution**: 
1. Ensure Gurobi is installed and licensed:
```bash
gurobi.sh
```

2. Verify Python can access Gurobi:
```bash
python -c "import gurobipy; print('Gurobi installed successfully')"
```

3. Install/update Gurobi Python bindings:
```bash
pip install --upgrade gurobipy
```

4. Restart Jupyter kernel after installing Gurobi

### Missing Data Files
Ensure all CSV files and JSON data are in the correct directory. The notebook automatically loads data via:
```python
from SystemCharacteristics import get_fixed_data
```

### Gurobi License Issues
If you get a license error:
```
Gurobi license check failed
```
**Solution**:
- Activate your Gurobi license (includes free academic licenses)
- Visit https://www.gurobi.com for license information
- Set `GRB_LICENSE_FILE` environment variable if using non-standard license location

### Memory Issues with 100 Days
If running out of memory, process in chunks:
```python
for batch in range(0, num_days, 25):  # Process 25 days at a time
    # ... optimization code
```

## References

- **Pyomo Documentation**: https://pyomo.readthedocs.io/
- **MILP Optimization**: Linear Programming and Mixed-Integer Programming fundamentals
- **HVAC Control**: Building energy management and comfort optimization
- **DTU Course**: Decision Making Under Uncertainty, Spring 2026

## Author & Contributors

Created for DTU Decision-Making Under Uncertainty course.

## License & Usage

This is an academic assignment for educational purposes. Use and modification are permitted for educational contexts.

## Related Files

- **Original Assignment**: See `Assignment_2026 Part A.pdf` for full task specifications
- **System Parameters**: Refer to `SystemCharacteristics.py` for all numerical values
- **Results Storage**: Solution data stored as dictionaries with keys: `['objective_cost', 'v', 'p1', 'p2', 'T1', 'T2', 'H']`

---

**Last Updated**: March 2026  
**Status**: Active Development
