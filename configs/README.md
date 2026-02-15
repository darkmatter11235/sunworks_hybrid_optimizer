# Example Configuration Files

This directory contains sample configuration files for different scenarios.

## Files

### example_base.json
Base configuration matching the validated `oa_hybrid.xlsx` case:
- Solar: 830 MW
- Wind: 0 WTG
- BESS: 50 MW / 200 MWh
- Load: 800 MW

### example_high_solar.json
High solar capacity scenario:
- Solar: 1200 MW (increased)
- Wind: 0 WTG
- BESS: 75 MW / 300 MWh (increased for higher capacity)
- Load: 800 MW

### example_hybrid_with_wind.json
Hybrid solar + wind scenario:
- Solar: 600 MW
- Wind: 20 WTG
- BESS: 50 MW / 200 MWh
- Load: 800 MW

## Using These Configs

### With Web UI
1. Start web interface: `streamlit run app.py`
2. Manually adjust sliders to match values in these files
3. Or load in code (future feature)

### With CLI

```bash
# Run simulation with a config
python cli.py simulate --config configs/example_base.json

# Save results
python cli.py simulate \
    --config configs/example_high_solar.json \
    --output results/high_solar_results.csv \
    --summary results/high_solar_summary.json

# Run all configs
for config in configs/*.json; do
    python cli.py simulate --config "$config" --output "results/$(basename $config .json).csv"
done
```

### Windows Batch Script

```batch
@echo off
call venv\Scripts\activate

for %%f in (configs\*.json) do (
    echo Running %%~nf...
    python cli.py simulate --config %%f --output results\%%~nf.csv
)

echo All scenarios complete!
pause
```

## Creating Custom Configs

1. Copy an example config:
   ```bash
   cp configs/example_base.json configs/my_scenario.json
   ```

2. Edit parameters as needed
3. Run simulation with your config

## Parameter Reference

| Parameter | Description | Units | Typical Range |
|-----------|-------------|-------|---------------|
| `solar_ac_mw` | Solar PV capacity | MW AC | 0-2000 |
| `dc_ac_ratio` | Solar DC/AC ratio | - | 1.0-2.0 |
| `wind_wtg_count` | Number of wind turbines | count | 0-100 |
| `bess_power_mw` | BESS power rating | MW | 0-200 |
| `bess_energy_mwh` | BESS energy capacity | MWh | 0-1000 |
| `total_load_mw` | Base load | MW | 100-5000 |
| `contracted_demand_mw` | Contracted demand | MW | = load |
| `evac_limit_mw` | Evacuation limit | MW | >= load |
| `solar_degrad` | Solar degradation | /year | 0.003-0.01 |
| `wind_degrad` | Wind degradation | /year | 0.003-0.01 |
| `one_way_eff` | BESS one-way efficiency | - | 0.85-0.95 |
| `tl_loss` | Transmission loss | fraction | 0.01-0.10 |
| `wheeling_loss` | Wheeling loss | fraction | 0-0.05 |
| `wind_expected_cuf` | Expected wind CUF | - | 0.25-0.45 |
| `wind_reference_cuf` | Reference wind CUF | - | 0.25-0.45 |
| `p_multiplier` | Profile multiplier | - | 0.8-1.2 |
| `penalty_per_unit_short` | Penalty rate | ₹/kWh | 0-10 |
| `normal_rate` | Normal TOD rate | ₹/kWh | 4-7 |
| `peak_rate` | Peak TOD rate | ₹/kWh | 5-10 |
| `offpeak_rate` | Off-peak TOD rate | ₹/kWh | 3-6 |
| `grid_available` | Grid availability | fraction | 0.95-1.0 |

## Tips

- **Start simple**: Modify one parameter at a time
- **Validate range**: Keep parameters within realistic bounds
- **Check consistency**: Ensure evac_limit >= load, BESS energy >= power
- **Name clearly**: Use descriptive names like `high_wind_scenario.json`
- **Document changes**: Add comments in a separate notes file
