# Sunworks Hybrid Optimizer

A tool for simulating and optimising hybrid renewable energy systems — solar, wind, and battery storage (BESS) — and calculating the Levelised Cost of Energy (LCOE).

---

## Table of Contents

1. [What Does This Tool Do?](#what-does-this-tool-do)
2. [Setup — First Time Only](#setup--first-time-only)
3. [Running a Simulation](#running-a-simulation)
4. [Ready-Made Scenarios](#ready-made-scenarios)
5. [Creating Your Own Scenario](#creating-your-own-scenario)
6. [Excel Template for Generation Profiles](#excel-template-for-generation-profiles)
7. [Running the Optimizer](#running-the-optimizer)
8. [Understanding the Output](#understanding-the-output)
9. [Parameter Reference](#parameter-reference)
10. [Project Structure (for developers)](#project-structure-for-developers)

---

## What Does This Tool Do?

Given a system design (how much solar, wind, and battery storage you have), this tool:

- Simulates every hour of a full year (8 760 hours) of generation, storage, and delivery
- Calculates how much energy was delivered to the grid vs. curtailed or imported
- Computes the **LCOE** (₹/kWh) — the true lifetime cost per unit of energy delivered
- Can automatically **search hundreds of designs** to find the one with the lowest LCOE

---

## Setup — First Time Only

You only need to do this once.

### 1. Make sure Python is installed

Open a terminal and type:

```bash
python --version
```

You need **Python 3.11 or newer**. If it's missing, download it from [python.org](https://www.python.org/downloads/).

### 2. Create and activate the virtual environment

```bash
cd /path/to/sunworks_hybrid_optimizer   # navigate to the project folder
python -m venv .venv                    # create the environment
source .venv/bin/activate               # activate it (Mac/Linux)
.venv\Scripts\activate                  # activate it (Windows)
```

Your terminal prompt should now show `(.venv)` at the start.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Tip:** Every time you open a new terminal, run `source .venv/bin/activate` (Mac/Linux) or `.venv\Scripts\activate` (Windows) before using the tool.

---

## Running a Simulation

A simulation takes one scenario (a configuration file) and produces energy and cost results for the full year.

### Basic run

```bash
python cli.py simulate --config configs/example_base.json
```

You will see output like:

```
Configuration: Solar=830.0MW, Wind=0WTG, BESS=50.0MW/200.0MWh
Running simulation...

LCOE: ₹1.97/kWh

=== Simulation Results ===
Solar Generation:        2,013,792 MWh
Wind Generation:                 0 MWh
Total Generation:        2,013,792 MWh
BESS Charge:                38,182 MWh
BESS Discharge:             34,364 MWh
Curtailment:                24,781 MWh
Grid Import:             5,092,290 MWh
Direct Delivered:        1,882,549 MWh
From BESS Delivered:        33,161 MWh
Total Delivered:         1,915,710 MWh
```

### Save results to files

Add `--output` to save the full hourly breakdown as a CSV, and `--summary` to save a concise JSON summary:

```bash
python cli.py simulate \
    --config configs/example_base.json \
    --output results/base_hourly.csv \
    --summary results/base_summary.json
```

> Create a `results/` folder first if it does not exist: `mkdir results`

---

## Ready-Made Scenarios

Three example configurations are included in the `configs/` folder:

| File                            | Solar    | Wind        | BESS            | Description                               |
| ------------------------------- | -------- | ----------- | --------------- | ----------------------------------------- |
| `example_base.json`             | 830 MW   | —           | 50 MW / 200 MWh | Baseline solar-only system                |
| `example_high_solar.json`       | 1 200 MW | —           | 75 MW / 300 MWh | Oversized solar to reduce grid dependency |
| `example_hybrid_with_wind.json` | 600 MW   | 20 turbines | 50 MW / 200 MWh | Solar + wind hybrid                       |

Run all three and compare LCOE in one go:

```bash
mkdir -p results

python cli.py simulate --config configs/example_base.json \
    --summary results/summary_base.json

python cli.py simulate --config configs/example_high_solar.json \
    --summary results/summary_high_solar.json

python cli.py simulate --config configs/example_hybrid_with_wind.json \
    --summary results/summary_hybrid.json
```

---

## Creating Your Own Scenario

1. Copy the closest example:

   ```bash
   cp configs/example_base.json configs/my_scenario.json
   ```

2. Open `configs/my_scenario.json` in any text editor and change the values you want. The most commonly adjusted parameters are:

   | Parameter         | What it controls                    | Example  |
   | ----------------- | ----------------------------------- | -------- |
   | `solar_ac_mw`     | Solar plant size (MW)               | `1000.0` |
   | `wind_wtg_count`  | Number of wind turbines             | `10`     |
   | `bess_power_mw`   | Battery discharge power (MW)        | `100.0`  |
   | `bess_energy_mwh` | Battery storage capacity (MWh)      | `400.0`  |
   | `total_load_mw`   | Contracted load you must serve (MW) | `800.0`  |
   | `evac_limit_mw`   | Maximum grid export capacity (MW)   | `800.0`  |

3. Run your scenario:

   ```bash
   python cli.py simulate --config configs/my_scenario.json
   ```

> See the full [Parameter Reference](#parameter-reference) section below for every available setting.

---

## Excel Template for Generation Profiles

If you want to maintain generation input data in Excel and support multiple `dc_ac_ratio` values, use:

```bash
python utils/generation_profiles_template.py create-template \
    --output standalone_data/generation_profiles_template.xlsx \
    --ratios 1.4 1.45 1.5
```

This creates an Excel workbook with:

- `meta` sheet: format notes
- `instructions` sheet: step-by-step data filling guidance
- `plant_info` sheet: plant/site metadata (name, place, latitude, longitude)
- `hourly_profiles` sheet: 8760 hourly rows with required columns

Required base columns in `hourly_profiles`:

- `hour_index`
- `hour_of_day`
- `wind_kwh_per_wtg`

Solar profile columns are one per ratio, using this naming format:

- `solar_kwh_per_mw_dcac_1.4`
- `solar_kwh_per_mw_dcac_1.45`
- `solar_kwh_per_mw_dcac_1.5`

After filling the sheet, convert Excel to simulator-ready CSV files:

```bash
python utils/generation_profiles_template.py convert \
    --input standalone_data/generation_profiles_template.xlsx \
    --output-dir standalone_data
```

This writes one CSV per ratio, such as:

- `generation_profiles_dcac_1_4.csv`
- `generation_profiles_dcac_1_45.csv`

And it also writes `generation_profiles.csv` as a default compatibility file.

If `plant_info` is filled, conversion also writes:

- `plant_location.json` (place name, latitude, longitude, and optional fields)

The simulator now auto-selects the right profile file based on `dc_ac_ratio` from your config.

---

## Running the Optimizer

The optimizer automatically tests many combinations of solar, wind, and BESS sizes and finds the one with the **lowest LCOE**.

### Basic optimize run

```bash
python cli.py optimize \
    --config standalone_data/system_config.json \
    --solar-min 600 --solar-max 1200 --solar-step 100 \
    --wind-min 0  --wind-max 20   --wind-step 5 \
    --output results/optimization.csv
```

This will test (1200−600)/100 + 1 = 7 solar sizes × (20−0)/5 + 1 = 5 wind counts = **35 combinations** and report the best one.

### Include BESS in the search

```bash
python cli.py optimize \
    --config standalone_data/system_config.json \
    --solar-min 600  --solar-max 1200 --solar-step 100 \
    --wind-min 0     --wind-max 20    --wind-step 5 \
    --bess-power-min 50  --bess-power-max 150 \
    --bess-energy-min 200 --bess-energy-max 600 \
    --output results/optimization_with_bess.csv
```

### Limit the number of configurations

If you want a quick exploratory run:

```bash
python cli.py optimize \
    --config standalone_data/system_config.json \
    --solar-min 500 --solar-max 1500 --solar-step 50 \
    --wind-min 0    --wind-max 30    --wind-step 5 \
    --max-configs 50 \
    --output results/quick_opt.csv
```

The results CSV contains every combination tested, sorted so you can open it in Excel and sort by `lcoe_inr_per_kwh` to find the best design.

---

## Understanding the Output

### Simulation results

| Field                       | Meaning                                                                             |
| --------------------------- | ----------------------------------------------------------------------------------- |
| **LCOE**                    | Levelised Cost of Energy — the average cost per kWh over the project's 25-year life |
| **Solar / Wind Generation** | Total energy produced in MWh over the year                                          |
| **BESS Charge / Discharge** | Energy put into / taken out of the battery                                          |
| **Curtailment**             | Energy that was generated but had to be thrown away (grid was full)                 |
| **Grid Import**             | Energy bought from the grid to supplement generation                                |
| **Direct Delivered**        | Renewable energy sent straight to the load without going through storage            |
| **From BESS Delivered**     | Energy served from the battery                                                      |
| **Total Delivered**         | Everything actually served to the load                                              |

### Summary JSON

The `--summary` file is a machine-readable snapshot useful for comparing many runs. Open it in any text editor or load it into Excel/Python.

### Optimization CSV

Each row is one configuration. Key columns:

| Column                              | Meaning                                   |
| ----------------------------------- | ----------------------------------------- |
| `solar_ac_mw`                       | Solar size tested                         |
| `wind_wtg_count`                    | Wind turbines tested                      |
| `bess_power_mw` / `bess_energy_mwh` | Battery size tested                       |
| `lcoe_inr_per_kwh`                  | LCOE for this combination                 |
| `is_feasible`                       | Whether this design meets all constraints |

Sort by `lcoe_inr_per_kwh` (ascending) to find the optimal design.

---

## Parameter Reference

Full list of settings available in a config file:

| Parameter               | Description                                  | Units         | Typical Range  |
| ----------------------- | -------------------------------------------- | ------------- | -------------- |
| `solar_ac_mw`           | Solar PV AC capacity                         | MW            | 200 – 2 000    |
| `dc_ac_ratio`           | Solar DC oversizing ratio                    | —             | 1.2 – 1.6      |
| `wind_wtg_count`        | Number of wind turbines                      | count         | 0 – 100        |
| `wind_capacity_per_wtg` | Rated power per turbine                      | MW            | 3.0 – 5.0      |
| `bess_power_mw`         | Battery power rating                         | MW            | 0 – 500        |
| `bess_energy_mwh`       | Battery energy capacity                      | MWh           | 0 – 2 000      |
| `bess_mode`             | Battery dispatch mode (`PV_SHIFT` or `FDRE`) | —             | —              |
| `total_load_mw`         | Peak load to serve                           | MW            | 100 – 5 000    |
| `contracted_demand_mw`  | Contracted demand (usually = load)           | MW            | = load         |
| `evac_limit_mw`         | Maximum export to grid                       | MW            | ≥ load         |
| `solar_degrad`          | Annual solar degradation                     | fraction/year | 0.003 – 0.010  |
| `wind_degrad`           | Annual wind degradation                      | fraction/year | 0.003 – 0.010  |
| `one_way_eff`           | One-way battery efficiency                   | fraction      | 0.85 – 0.97    |
| `tl_loss`               | Transmission line loss                       | fraction      | 0.01 – 0.10    |
| `wheeling_loss`         | Wheeling loss (third-party transmission)     | fraction      | 0 – 0.05       |
| `wind_expected_cuf`     | Expected wind capacity utilisation factor    | fraction      | 0.25 – 0.45    |
| `p_multiplier`          | Generation profile scaling factor            | —             | 0.8 – 1.2      |
| `normal_rate`           | Normal TOD tariff                            | ₹/kWh         | 4 – 7          |
| `peak_rate`             | Peak TOD tariff                              | ₹/kWh         | 5 – 10         |
| `offpeak_rate`          | Off-peak TOD tariff                          | ₹/kWh         | 3 – 6          |
| `grid_available`        | Grid availability fraction                   | fraction      | 0.95 – 1.0     |
| `banking_enabled`       | Enable quarterly energy banking              | bool          | `true`/`false` |

> **TOD Schedule** — Normal: 00:00–05:00, 17:00–19:00, 23:00–00:00 · Peak: 05:00–09:00, 19:00–23:00 · Off-peak: 09:00–17:00

---

## Project Structure (for developers)

```
cli.py                      ← Command-line entry point
hourly_sim_skeleton.py      ← 8 760-hour simulation engine
lcoe_calculator.py          ← NPV-based LCOE calculation
optimizer.py                ← Grid-search optimizer
app.py                      ← Streamlit web UI
banking_settlement_skeleton.py  ← Quarterly banking logic

configs/                    ← Ready-made scenario files
standalone_data/            ← Generation & load profiles + financial params
utils/                      ← Data extraction helpers
validation/                 ← Validation scripts against Excel baseline
```

### Key technical notes

- All energy values are in **MWh**, power in **MW**
- Hours are 0-indexed (hour 0 = 00:00–01:00)
- LCOE uses a 25-year project lifetime with NPV discounting, depreciation tax shields, and loan schedules
- Battery degradation and solar/wind degradation are applied year-on-year in the LCOE calculation
- The model has been validated to 0.000% error against the Excel baseline
