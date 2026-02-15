"""
Compare Python vs Excel LCOE calculation components
"""
import openpyxl
import numpy as np
import pandas as pd
from hourly_sim_skeleton import Config, load_profiles_from_workbook, simulate_hourly
from lcoe_calculator import (
    CostParameters, FinancialParameters, calculate_lcoe,
    calculate_depreciation_schedule, calculate_loan_schedule
)

# Load Excel
wb = openpyxl.load_workbook('oa_hybrid.xlsx', data_only=True)
ws_sum = wb['Summary']
ws_hd = wb[' Hourly Data']
ws_lcoe = wb['LCOE']

# Run Python simulation
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

df_profiles = load_profiles_from_workbook('oa_hybrid.xlsx', cfg)
df_sim = simulate_hourly(cfg, df_profiles)

# Excel values
excel_pci = float(ws_lcoe['E4'].value)   # Equity
excel_npv_opex = float(ws_lcoe['E5'].value)
excel_npv_residual = float(ws_lcoe['E6'].value)
excel_npv_dep = float(ws_lcoe['E7'].value)
excel_npv_energy = float(ws_lcoe['E12'].value)
excel_npv_interest = float(ws_lcoe['E17'].value)  
excel_npv_loan_payment = float(ws_lcoe['E18'].value)
excel_tax_rate = float(ws_lcoe['D9'].value)
excel_discount_rate = float(ws_lcoe['D11'].value)
excel_project_cost = float(ws_lcoe['D3'].value)
excel_annual_opex = float(ws_lcoe['D5'].value)

print("="*80)
print("EXCEL VALUES")
print("="*80)
print(f"Project Cost:         ₹{excel_project_cost:.2f} Crore")
print(f"Equity (30%):         ₹{excel_pci:.2f} Crore")
print(f"Annual OPEX:          ₹{excel_annual_opex:.2f} Crore")
print(f"Tax Rate:             {excel_tax_rate:.4f}")
print(f"Discount Rate:        {excel_discount_rate:.7f}")
print(f"\nNPV Components:")
print(f"  NPV Depreciation:   ₹{excel_npv_dep:.2f} Crore")
print(f"  NPV Interest:       ₹{excel_npv_interest:.2f} Crore")
print(f"  NPV Loan Payment:   ₹{excel_npv_loan_payment:.2f} Crore")
print(f"  NPV OPEX:           ₹{excel_npv_opex:.2f} Crore")
print(f"  NPV Residual:       ₹{excel_npv_residual:.2f} Crore")
print(f"  NPV Energy:         {excel_npv_energy:.2f} Crore kWh")

excel_lcoe = (excel_pci - (excel_npv_dep + excel_npv_interest)*excel_tax_rate + 
              excel_npv_loan_payment + excel_npv_opex*(1-excel_tax_rate) - excel_npv_residual) / excel_npv_energy
print(f"\nExcel LCOE:           ₹{excel_lcoe:.4f}/kWh")

# Python calculation
print("\n" + "="*80)
print("PYTHON CALCULATION")
print("="*80)

fin_params = FinancialParameters(
    tax_rate=excel_tax_rate,
    discount_rate=excel_discount_rate,
    interest_rate=0.095,
    system_degradation=cfg.solar_degrad,
    opex_escalation=0.05,  # 5% annual escalation
    equity_fraction=0.30,
    loan_fraction=0.70,
    loan_term_years=10,
    depreciation_rate_early=0.04666666,
    depreciation_rate_late=0.03,
    depreciation_switchover_year=15,
    project_lifetime_years=25
)

cost_params = CostParameters()

# Calculate schedules
depreciation = calculate_depreciation_schedule(excel_project_cost, fin_params)
principal_payments, interest_payments, outstanding = calculate_loan_schedule(excel_project_cost, fin_params)

print(f"Year 1 depreciation:  ₹{depreciation[0]:.2f} Crore")
print(f"Year 1 interest:      ₹{interest_payments[0]:.2f} Crore")
print(f"Year 1 principal:     ₹{principal_payments[0]:.2f} Crore")
print(f"Year 1 total payment: ₹{principal_payments[0] + interest_payments[0]:.2f} Crore")

#NPV calculations
dr = excel_discount_rate
python_npv_dep = sum(depreciation[year] / ((1 + dr) ** (year + 1)) for year in range(25))
python_npv_interest = sum(interest_payments[year] / ((1 + dr) ** (year + 1)) for year in range(25))
python_npv_loan_payment = sum((principal_payments[year] + interest_payments[year]) / ((1 + dr) ** (year + 1)) for year in range(25))
# OPEX NPV without tax (tax applied in final formula)
python_npv_opex = sum(excel_annual_opex * ((1 + 0.05) ** year) / ((1 + dr) ** (year + 1)) for year in range(25))

print(f"\nPython NPV components:")
print(f"  NPV Depreciation:   ₹{python_npv_dep:.2f} Crore")
print(f"  NPV Interest:       ₹{python_npv_interest:.2f} Crore")
print(f"  NPV Loan Payment:   ₹{python_npv_loan_payment:.2f} Crore")
print(f"  NPV OPEX:           ₹{python_npv_opex:.2f} Crore")

# Energy calculation
first_year_delivered_kwh = df_sim['delivered_total'].sum() * 1000  # MWh to kWh
first_year_delivered_crore_kwh = first_year_delivered_kwh / 1e7

print(f"\nYear 1 delivered:     {first_year_delivered_crore_kwh:.2f} Crore kWh")

# Calculate energy NPV with degradation
annual_energy_crore_kwh = []
for year in range(25):
    degraded = first_year_delivered_crore_kwh * ((1 - cfg.solar_degrad) ** year)
    annual_energy_crore_kwh.append(degraded)
    if year < 3:
        print(f"  Year {year+1}: {degraded:.2f} Crore kWh")

python_npv_energy = sum(annual_energy_crore_kwh[year] / ((1 + dr) ** (year + 1)) for year in range(25))
print(f"\nPython NPV Energy:    {python_npv_energy:.2f} Crore kWh")

# Calculate Python LCOE using Excel formula (apply tax to OPEX here)
python_lcoe = (excel_pci - (python_npv_dep + python_npv_interest)*excel_tax_rate + 
               python_npv_loan_payment + python_npv_opex*(1-excel_tax_rate) - 0) / python_npv_energy

print(f"Python LCOE:          ₹{python_lcoe:.4f}/kWh")

# Comparison
print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"{'Component':<30} {'Excel':<15} {'Python':<15} {'Diff %':<10}")
print("-"*80)
print(f"{'NPV Depreciation':<30} {excel_npv_dep:>14.2f} {python_npv_dep:>14.2f} {100*(python_npv_dep-excel_npv_dep)/excel_npv_dep:>9.2f}%")
print(f"{'NPV Interest':<30} {excel_npv_interest:>14.2f} {python_npv_interest:>14.2f} {100*(python_npv_interest-excel_npv_interest)/excel_npv_interest:>9.2f}%")
print(f"{'NPV Loan Payment':<30} {excel_npv_loan_payment:>14.2f} {python_npv_loan_payment:>14.2f} {100*(python_npv_loan_payment-excel_npv_loan_payment)/excel_npv_loan_payment:>9.2f}%")
print(f"{'NPV OPEX':<30} {excel_npv_opex:>14.2f} {python_npv_opex:>14.2f} {100*(python_npv_opex-excel_npv_opex)/excel_npv_opex:>9.2f}%")
print(f"{'NPV Energy':<30} {excel_npv_energy:>14.2f} {python_npv_energy:>14.2f} {100*(python_npv_energy-excel_npv_energy)/excel_npv_energy:>9.2f}%")
print(f"{'LCOE':<30} {excel_lcoe:>14.4f} {python_lcoe:>14.4f} {100*(python_lcoe-excel_lcoe)/excel_lcoe:>9.2f}%")
