#!/usr/bin/env python3
"""
Compare LCOE calculation between Excel and Python implementation.
"""
import json
import pandas as pd
import openpyxl
from hourly_sim_skeleton import Config, load_profiles_from_workbook, simulate_hourly
from lcoe_calculator import calculate_lcoe

def compare_lcoe(xlsx_path, data_dir="standalone_data"):
    """Compare LCOE from Excel workbook vs Python calculation."""
    
    print("="*70)
    print("COMPARING LCOE: EXCEL vs PYTHON")
    print("="*70)
    
    # Load Excel workbook
    print("\n1. Loading Excel workbook...")
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws_sum = wb['Summary']
    ws_hd = wb[' Hourly Data']
    ws_lcoe = wb['LCOE']
    
    # Extract LCOE from Excel
    # B19: LCOE with AD per kWh (with accelerated depreciation)
    # B20: LCOE per kWh (without AD)
    # B42: Levelized Tariff
    excel_lcoe = float(ws_lcoe['B20'].value)  # LCOE in ₹/kWh (without AD)
    excel_lcoe_with_ad = float(ws_lcoe['B19'].value)  # LCOE with AD
    print(f"   Excel LCOE (B20, without AD): ₹{excel_lcoe:.4f}/kWh")
    print(f"   Excel LCOE (B19, with AD):    ₹{excel_lcoe_with_ad:.4f}/kWh")
    
    # Run Python simulation
    print("\n2. Running Python simulation...")
    cfg = Config(
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
    
    df_profiles = load_profiles_from_workbook(xlsx_path, cfg)
    df_sim = simulate_hourly(cfg, df_profiles)
    print(f"   Simulation complete: {len(df_sim)} hours")
    
    # Calculate Python LCOE
    print("\n3. Calculating Python LCOE using Excel cost parameters...")
    
    # Import required classes
    from lcoe_calculator import (
        CostParameters, FinancialParameters, calculate_lcoe
    )
    import numpy as np
    
    # Extract actual costs from Excel instead of calculating
    project_cost_crore = float(ws_lcoe['D3'].value)  # Total project cost from Excel
    annual_opex_crore = float(ws_lcoe['D5'].value)   # Annual OPEX from Excel
    excel_tax_rate = float(ws_lcoe['D9'].value)
    
   # Extract Excel NPV values for comparison
    excel_npv_dep = float(ws_lcoe['E7'].value)
    excel_npv_interest = float(ws_lcoe['E17'].value)
    excel_npv_loan_payment = float(ws_lcoe['E18'].value)
    excel_npv_opex = float(ws_lcoe['E5'].value)
    
    print(f"   Excel Project Cost: ₹{project_cost_crore:.2f} Crore")
    print(f"   Excel Annual OPEX:  ₹{annual_opex_crore:.2f} Crore")
    
    # Use Excel financial parameters
    fin_params = FinancialParameters(
        tax_rate=float(ws_lcoe['D9'].value),
        discount_rate=float(ws_lcoe['D11'].value),
        interest_rate=0.095,  # Default, not critical for calculation
        system_degradation=cfg.solar_degrad,
        opex_escalation=0.05,  # 5% annual OPEX escalation
        equity_fraction=float(ws_lcoe['C4'].value),
        loan_fraction=float(ws_lcoe['B15'].value),
        loan_term_years=int(ws_lcoe['B16'].value),
        depreciation_rate_early=0.04666666,
        depreciation_rate_late=0.03,
        depreciation_switchover_year=15,
        project_lifetime_years=25
    )
    
    # Dummy cost params (set residual to 0 to match Excel)
    cost_params = CostParameters(residual_value_fraction=0.0)
    
    # Calculate annual energy delivered in kWh
    first_year_delivered_kwh = df_sim['delivered_total'].sum() * 1000  # MWh to kWh
    # Pass constant energy - calculate_lcoe will apply degradation internally
    annual_energy_kwh = np.full(fin_params.project_lifetime_years, first_year_delivered_kwh)
    
    print(f"   First Year Energy:  {first_year_delivered_kwh/1e9:.2f} GWh ({first_year_delivered_kwh/1e7:.2f} Crore kWh)")
    
    # Calculate LCOE
    results = calculate_lcoe(project_cost_crore, annual_opex_crore, annual_energy_kwh, fin_params, cost_params)
    python_lcoe = results['lcoe_inr_per_kwh']
    
    print(f"   Python LCOE:        ₹{python_lcoe:.4f}/kWh")
    print(f"\n   NPV Component Comparison:")
    print(f"                        {'Excel':>12} {'Python':>12} {'Diff':>12}")
    print(f"     OPEX:             {excel_npv_opex:>12.2f} {results['opex_npv_crore']:>12.2f} {results['opex_npv_crore']-excel_npv_opex:>12.2f}")
    print(f"     Tax Shield:       {(excel_npv_dep+excel_npv_interest)*excel_tax_rate:>12.2f} {results['tax_shield_npv_crore']:>12.2f} {results['tax_shield_npv_crore']-(excel_npv_dep+excel_npv_interest)*excel_tax_rate:>12.2f}")
    print(f"     Loan Payment:     {excel_npv_loan_payment:>12.2f} {results['loan_payment_npv_crore']:>12.2f} {results['loan_payment_npv_crore']-excel_npv_loan_payment:>12.2f}")
    
    # Calculate expected cost NPV
    print(f"     Equity:           {1184.87:>12.2f} {results['equity_crore']:>12.2f} {results['equity_crore']-1184.87:>12.2f}")
    expected_cost_npv = results['equity_crore'] - results['tax_shield_npv_crore'] + results['loan_payment_npv_crore'] + results['opex_npv_crore'] * (1 - excel_tax_rate) - results['residual_npv_crore']
    print(f"     Expected Cost NPV:{expected_cost_npv:>12.2f} (= {results['equity_crore']:.2f} - {results['tax_shield_npv_crore']:.2f} + {results['loan_payment_npv_crore']:.2f} + {results['opex_npv_crore']*(1-excel_tax_rate):.2f} - {results['residual_npv_crore']:.2f})")
    print(f"     Total Cost NPV:   {3942.95:>12.2f} {results['total_cost_npv_crore']:>12.2f} {results['total_cost_npv_crore']-3942.95:>12.2f}")
    
    # Compare
    print("\n" + "="*70)
    print("LCOE COMPARISON")
    print("="*70)
    
    diff = python_lcoe - excel_lcoe
    diff_pct = 100 * diff / excel_lcoe if excel_lcoe != 0 else 0
    
    print(f"\n{'Source':<20} {'LCOE (₹/kWh)':<20} {'Difference':<20}")
    print("-"*70)
    print(f"{'Excel':<20} {excel_lcoe:>19.4f}")
    print(f"{'Python':<20} {python_lcoe:>19.4f} {diff:>19.4f} ({diff_pct:+.3f}%)")
    
    print("\n" + "="*70)
    if abs(diff_pct) < 0.01:  # Within 0.01%
        print("✓ SUCCESS - LCOE values match perfectly!")
    elif abs(diff_pct) < 0.1:  # Within 0.1%
        print("✓ SUCCESS - LCOE values match within 0.1%!")
    elif abs(diff_pct) < 1.0:  # Within 1%
        print("✓ CLOSE - LCOE values match within 1%")
    else:
        print("⚠ DIFFERENCE - LCOE values differ by more than 1%")
    print("="*70)
    
    # Show some intermediate calculations for debugging
    print("\n" + "="*70)
    print("ANNUAL GENERATION (Year 1)")
    print("="*70)
    total_delivered = df_sim['delivered_total'].sum()
    print(f"Total Energy Delivered: {total_delivered:,.2f} MWh")
    print(f"                        {total_delivered * 1000:,.2f} kWh")
    
    return excel_lcoe, python_lcoe, diff_pct

if __name__ == "__main__":
    import sys
    xlsx = sys.argv[1] if len(sys.argv) > 1 else "oa_hybrid.xlsx"
    data = sys.argv[2] if len(sys.argv) > 2 else "standalone_data"
    
    excel_lcoe, python_lcoe, diff_pct = compare_lcoe(xlsx, data)
    
    if abs(diff_pct) > 1.0:
        sys.exit(1)
    else:
        sys.exit(0)
