#!/usr/bin/env python3
"""
Run hourly simulation using standalone CSV/JSON files (no Excel).
Demonstrates that Python model can run completely independently.
"""
import json
import pandas as pd
import numpy as np
from hourly_sim_skeleton import Config, simulate_hourly

def run_standalone_simulation(data_dir="standalone_data"):
    """
    Run simulation using only CSV/JSON files.
    
    Required files in data_dir:
      - system_config.json
      - generation_profiles.csv
      - load_profile.csv (optional if load is constant)
      - financial_params.json (for LCOE calculation)
    """
    
    print("="*70)
    print("RUNNING STANDALONE SIMULATION (NO EXCEL)")
    print("="*70)
    
    # 1. Load system configuration
    print("\n1. Loading system configuration...")
    with open(f"{data_dir}/system_config.json") as f:
        config_dict = json.load(f)
    
    # Filter to only fields that Config accepts
    from dataclasses import fields
    valid_fields = {f.name for f in fields(Config)}
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
    
    cfg = Config(**filtered_config)
    print(f"   Solar: {cfg.solar_ac_mw} MW AC")
    print(f"   Wind: {cfg.wind_wtg_count} WTGs")
    print(f"   BESS: {cfg.bess_power_mw} MW / {cfg.bess_energy_mwh} MWh")
    print(f"   Load: {cfg.total_load_mw} MW")
    
    # 2. Load generation profiles
    print("\n2. Loading generation profiles...")
    df_gen = pd.read_csv(f"{data_dir}/generation_profiles.csv")
    print(f"   {len(df_gen)} hours loaded")
    print(f"   Solar: {df_gen['solar_kwh_per_mw'].sum():,.0f} kWh/MW")
    print(f"   Wind: {df_gen['wind_kwh_per_wtg'].sum():,.0f} kWh/WTG")
    
    # 3. Load load profile
    print("\n3. Loading load profile...")
    df_load = pd.read_csv(f"{data_dir}/load_profile.csv")
    print(f"   {len(df_load)} hours loaded")
    print(f"   Total load: {df_load['load_mw'].sum():,.0f} MWh")
    
    # 4. Prepare profiles dataframe for simulation
    print("\n4. Preparing simulation inputs...")
    
    # Create datetime index
    dt = pd.date_range('2024-01-01', periods=8760, freq='h')
    
    # Use hour_of_day from CSV if available, otherwise use datetime hour
    if 'hour_of_day' in df_gen.columns:
        hour = df_gen['hour_of_day'].values
    else:
        hour = dt.hour.to_numpy()
    
    month = dt.month.to_numpy()
    day = dt.day.to_numpy()
    
    # Assign TOD (replicate the assign_tod function logic)
    from hourly_sim_skeleton import assign_tod
    tod = assign_tod(hour)
    
    df_profiles = pd.DataFrame({
        'dt': dt,
        'month': month,
        'day': day,
        'hour': hour,
        'tod': tod,
        'solar_mwh_per_mw': df_gen['solar_kwh_per_mw'].values / 1000.0,
        'wind_mwh_per_wtg': df_gen['wind_kwh_per_wtg'].values / 1000.0,
        'load_mw': df_load['load_mw'].values,
    })
    
    print(f"   Profiles DataFrame: {len(df_profiles)} rows, {len(df_profiles.columns)} columns")
    
    # 5. Run hourly simulation
    print("\n5. Running hourly simulation...")
    df_sim = simulate_hourly(cfg, df_profiles)
    print(f"   Simulation complete: {len(df_sim)} hours")
    
    # 6. Display results
    print("\n" + "="*70)
    print("SIMULATION RESULTS")
    print("="*70)
    
    results = {
        "Solar Generation": df_sim['solar_gen'].sum(),
        "Wind Generation": df_sim['wind_gen'].sum(),
        "Total Generation": df_sim['gen_total'].sum(),
        "To Consumption": df_sim['to_consumption_inj'].sum(),
        "To Banking": df_sim['to_banking_inj'].sum(),
        "Curtailment": df_sim['curtailed'].sum(),
        "BESS Charge": df_sim['bess_charge_inj'].sum(),
        "BESS Discharge": df_sim['bess_discharge_inj'].sum(),
        "Direct Delivered": df_sim['direct_delivered'].sum(),
        "From BESS": df_sim['from_bess_delivered'].sum(),
        "Grid Import": df_sim['from_grid'].sum(),
        "Total Delivered": df_sim['delivered_total'].sum(),
    }
    
    print(f"\n{'Metric':<30} {'Value (MWh)':<20}")
    print("-"*70)
    for metric, value in results.items():
        print(f"{metric:<30} {value:>19,.2f}")
    
    # 7. Sample hourly data
    print("\n" + "="*70)
    print("SAMPLE HOURLY DATA (first 10 hours)")
    print("="*70)
    
    cols_to_show = ['hour', 'tod', 'solar_gen', 'wind_gen', 'load_total', 
                    'bess_charge_inj', 'bess_discharge_inj', 'soc_mwh', 'from_grid']
    print(df_sim[cols_to_show].head(10).to_string(index=False))
    
    print("\n" + "="*70)
    print("SUCCESS - Simulation ran without Excel!")
    print("="*70)
    
    return df_sim, results

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "standalone_data"
    df_sim, results = run_standalone_simulation(data_dir)
