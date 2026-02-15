#!/usr/bin/env python3
"""
Compare Excel-based simulation vs standalone CSV-based simulation.
"""
import json
import pandas as pd
from hourly_sim_skeleton import Config, load_profiles_from_workbook, simulate_hourly

def compare_excel_vs_standalone(xlsx_path, data_dir="standalone_data"):
    """Compare results from Excel vs standalone data."""
    
    print("="*70)
    print("COMPARING EXCEL vs STANDALONE SIMULATIONS")
    print("="*70)
    
    # Run Excel-based simulation
    print("\n1. Running Excel-based simulation...")
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws_sum = wb['Summary']
    ws_hd = wb[' Hourly Data']
    
    cfg_excel = Config(
        year=1,
        total_load_mw=float(ws_sum['B8'].value),
        contracted_demand_mw=float(ws_hd['B4'].value),
        evac_limit_mw=float(ws_hd['B3'].value),
        tl_loss=float(ws_sum['S9'].value or 0.0),
        wheeling_loss=float(ws_sum['S10'].value or 0.0),
        dc_ac_ratio=float(ws_sum['B13'].value),
        solar_ac_mw=float(ws_sum['B14'].value),
        wind_wtg_count=int(ws_sum['B21'].value or 0),
        wind_expected_cuf=float(ws_sum['B23'].value or 0.32),
        wind_reference_cuf=float(ws_sum['B24'].value or 0.355587),
        solar_degrad=float(ws_sum['B19'].value or 0.005),
        wind_degrad=float(ws_sum['B25'].value or 0.005),
        one_way_eff=float(ws_hd['B7'].value),
        bess_power_mw=float(ws_hd['B5'].value),
        bess_energy_mwh=float(ws_hd['B8'].value),
        bess_mode="PV_SHIFT",
    )
    
    df_profiles_excel = load_profiles_from_workbook(xlsx_path, cfg_excel)
    df_sim_excel = simulate_hourly(cfg_excel, df_profiles_excel)
    
    print(f"   Excel: {len(df_sim_excel)} hours simulated")
    
    # Run standalone simulation
    print("\n2. Running standalone simulation...")
    with open(f"{data_dir}/system_config.json") as f:
        config_dict = json.load(f)
    
    from dataclasses import fields
    valid_fields = {f.name for f in fields(Config)}
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
    cfg_standalone = Config(**filtered_config)
    
    df_gen = pd.read_csv(f"{data_dir}/generation_profiles.csv")
    df_load = pd.read_csv(f"{data_dir}/load_profile.csv")
    
    dt = pd.date_range('2024-01-01', periods=8760, freq='h')
    
    # Use hour_of_day from CSV if available
    if 'hour_of_day' in df_gen.columns:
        hour = df_gen['hour_of_day'].values
    else:
        hour = dt.hour.to_numpy()
    
    from hourly_sim_skeleton import assign_tod
    tod = assign_tod(hour)
    
    df_profiles_standalone = pd.DataFrame({
        'dt': dt,
        'month': dt.month.to_numpy(),
        'day': dt.day.to_numpy(),
        'hour': hour,
        'tod': tod,
        'solar_mwh_per_mw': df_gen['solar_kwh_per_mw'].values / 1000.0,
        'wind_mwh_per_wtg': df_gen['wind_kwh_per_wtg'].values / 1000.0,
        'load_mw': df_load['load_mw'].values,
    })
    
    df_sim_standalone = simulate_hourly(cfg_standalone, df_profiles_standalone)
    print(f"   Standalone: {len(df_sim_standalone)} hours simulated")
    
    # Compare results
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    metrics = [
        ('Solar Generation', 'solar_gen'),
        ('Wind Generation', 'wind_gen'),
        ('Total Generation', 'gen_total'),
        ('To Consumption', 'to_consumption_inj'),
        ('BESS Charge', 'bess_charge_inj'),
        ('BESS Discharge', 'bess_discharge_inj'),
        ('Curtailment', 'curtailed'),
        ('Direct Delivered', 'direct_delivered'),
        ('From BESS', 'from_bess_delivered'),
        ('Grid Import', 'from_grid'),
    ]
    
    print(f"\n{'Metric':<25} {'Excel (MWh)':<18} {'Standalone (MWh)':<18} {'Diff %':<12}")
    print("-"*73)
    
    for label, col in metrics:
        excel_val = df_sim_excel[col].sum()
        standalone_val = df_sim_standalone[col].sum()
        diff_pct = 100 * (standalone_val - excel_val) / excel_val if excel_val != 0 else 0
        print(f"{label:<25} {excel_val:>17,.2f} {standalone_val:>17,.2f} {diff_pct:>11.3f}%")
    
    print("\n" + "="*70)
    if all(abs(100 * (df_sim_standalone[col].sum() - df_sim_excel[col].sum()) / 
               df_sim_excel[col].sum() if df_sim_excel[col].sum() != 0 else 0) < 0.01 
           for _, col in metrics):
        print("✓ SUCCESS - Excel and standalone produce identical results!")
    else:
        print("⚠ MISMATCH - Some differences detected")
    print("="*70)

if __name__ == "__main__":
    import sys
    xlsx = sys.argv[1] if len(sys.argv) > 1 else "oa_hybrid.xlsx"
    data = sys.argv[2] if len(sys.argv) > 2 else "standalone_data"
    compare_excel_vs_standalone(xlsx, data)
