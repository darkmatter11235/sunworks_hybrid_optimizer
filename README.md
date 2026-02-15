# Sunworks Hybrid Optimizer

Python-based hybrid renewable energy system optimizer for solar/wind/BESS configurations with LCOE calculations.

## Project Structure

### Core Modules

- **`hourly_sim_skeleton.py`** - Main hourly simulation engine
  - Simulates 8760 hours of solar/wind generation
  - BESS charge/discharge logic (PV_SHIFT and FDRE modes)
  - Grid import/export calculations
  - TOD (time-of-day) period handling
  - Banking settlement (if enabled)

- **`lcoe_calculator.py`** - Levelized Cost of Energy calculator
  - NPV-based LCOE calculation with full financial modeling
  - Project cost calculations (solar, wind, BESS)
  - Depreciation, tax shields, and loan payments
  - Annual operating expenses
  - 25-year project lifetime

- **`optimizer.py`** - Optimization framework
  - Grid search over solar/wind/BESS configuration space
  - Finds optimal combination to minimize LCOE
  - Parallel evaluation support
  - Results export and visualization

- **`banking_settlement_skeleton.py`** - Banking settlement logic (future use)
  - Quarterly settlement calculations
  - Banking carry-forward logic
  - TOD-based energy accounting

- **`validate_hourly_model.py`** - Validation tool
  - Compares Python simulation against Excel baseline
  - Column-by-column validation with error reporting
  - Summary statistics comparison

### Configuration Files

- **`optimization_config.json`** - Optimization search space definition
  - Solar AC capacity range
  - BESS power/energy ranges  
  - Grid search step sizes
  - Financial parameters

- **`lcoe_params.json`** - LCOE calculation parameters
  - Project costs per unit
  - Tax rates, discount rates
  - O&M costs, insurance rates
  - Loan terms

- **`spec.json`** - Excel model specification
  - Sheet structure and column mappings
  - Formula documentation
  - Parameter locations

### Data

- **`oa_hybrid.xlsx`** - Excel baseline model
  - Source of truth for validation
  - Contains hourly generation profiles
  - LCOE calculations and parameters

### Documentation

- **`README_OPTIMIZATION.md`** - Optimization framework guide
- **`README_port_to_python.md`** - Excel to Python porting notes

### Utilities

- **`utils/`** - Extraction and utility scripts
  - `extract_excel_spec.py` - Extract Excel structure
  - `extract_lcoe_params.py` - Extract LCOE parameters
  - See [utils/README.md](utils/README.md)

### Debug

- **`debug/`** - Debugging and validation scripts
  - Historical debugging scripts from model development
  - Validation helpers and test scripts
  - See [debug/README.md](debug/README.md)

## Quick Start

### Validate Hourly Model

```tcsh
python validate_hourly_model.py oa_hybrid.xlsx
```

### Calculate LCOE

```python
from hourly_sim_skeleton import Config, load_profiles_from_workbook, simulate_hourly
from lcoe_calculator import calculate_lcoe

# Configure system
cfg = Config(
    solar_ac_mw=830,
    wind_wtg_count=0,
    bess_power_mw=50,
    bess_energy_mwh=200,
    # ... other params
)

# Simulate
df_profiles = load_profiles_from_workbook('oa_hybrid.xlsx', cfg)
df_sim = simulate_hourly(cfg, df_profiles)

# Calculate LCOE
lcoe = calculate_lcoe(cfg, df_sim, 'lcoe_params.json')
print(f"LCOE: ₹{lcoe:.4f}/kWh")
```

### Run Optimization

```python
from optimizer import optimize

results = optimize(
    xlsx_path='oa_hybrid.xlsx',
    config_path='optimization_config.json',
    output_path='optimization_results.csv'
)
```

## Model Validation Status

✅ **Fully Validated** - All hourly calculations match Excel with 0.000% error:
- Solar/Wind generation
- BESS charge/discharge
- Curtailment
- Grid import
- Direct consumption delivery
- TOD classifications

## Key Features

- **Accurate BESS modeling**: PV_SHIFT (load shifting) and FDRE (firm dispatch) modes
- **TOD-aware**: Peak/normal/offpeak period handling for settlement
- **Loss modeling**: Transmission and wheeling losses
- **Financial rigor**: Full NPV-based LCOE with depreciation and tax shields
- **Validated**: 100% match with Excel baseline model

## Requirements

- Python 3.11+
- pandas
- numpy
- openpyxl

Install dependencies:
```tcsh
python -m venv venv
source venv/bin/activate.csh  # tcsh
pip install pandas numpy openpyxl
```

## Notes

- Hours are 0-23 (hour 0 = 00:00-01:00, hour 23 = 23:00-00:00)
- Timestamps in Excel represent end-of-hour
- TOD Schedule: normal (0-4, 17-18, 23), peak (5-8, 19-22), offpeak (9-16)
- All energy values in MWh, power in MW
