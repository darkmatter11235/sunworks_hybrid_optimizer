#!/usr/bin/env python3
"""
LCOE Calculator Module

Implements the LCOE formula:
LCOE = [PCI - Σ(DEP+INT)×TR/(1+DR)^n + Σ LP/(1+DR)^n + Σ AO/(1+DR)^n×(1-TR) - RV/(1+DR)^n] 
       / [Σ Initial kWh × (1-SDR)^n / (1+DR)^n]

Where:
- PCI = Project Capital Investment (equity portion)
- DEP = Depreciation 
- INT = Interest on loan
- TR = Tax Rate
- LP = Loan Payment (principal + interest)
- AO = Annual Operating costs
- RV = Residual Value
- SDR = System Degradation Rate
- DR = Discount Rate
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class CostParameters:
    """Cost parameters for project components."""
    # Capital costs (Crore INR per unit)
    solar_capex_per_mw: float = 3.2  # Crore/MW AC
    wind_capex_per_mw: float = 6.0   # Crore/MW
    bess_power_capex_per_mw: float = 1.5  # Crore/MW
    bess_energy_capex_per_mwh: float = 2.0  # Crore/MWh
    
    # Operating costs (Crore INR per year)
    solar_opex_per_mw_year: float = 0.025  # Crore/MW/year
    wind_opex_per_mw_year: float = 0.045   # Crore/MW/year
    bess_opex_per_mwh_year: float = 0.010  # Crore/MWh/year
    fixed_opex: float = 5.0  # Crore/year (land, admin, etc)
    
    # Residual value (% of capital cost)
    residual_value_fraction: float = 0.10


@dataclass
class FinancialParameters:
    """Financial parameters for LCOE calculation."""
    # Rates
    tax_rate: float = 0.2782
    discount_rate: float = 0.0749997
    interest_rate: float = 0.095
    system_degradation: float = 0.005
    opex_escalation: float = 0.05  # 5% annual OPEX escalation
    
    # Loan structure
    equity_fraction: float = 0.30
    loan_fraction: float = 0.70
    loan_term_years: int = 10
    
    # Depreciation (% of depreciable asset per year)
    depreciation_rate_early: float = 0.04666666  # Years 1-15
    depreciation_rate_late: float = 0.03          # Years 16+
    depreciation_switchover_year: int = 15
    
    # Project lifetime
    project_lifetime_years: int = 25


def calculate_project_cost(solar_ac_mw: float, wind_mw: float, 
                          bess_power_mw: float, bess_energy_mwh: float, 
                          cost_params: CostParameters) -> float:
    """
    Calculate total project capital cost.
    
    Returns:
        Total cost in Crore INR
    """
    solar_cost = solar_ac_mw * cost_params.solar_capex_per_mw
    wind_cost = wind_mw * cost_params.wind_capex_per_mw
    bess_power_cost = bess_power_mw * cost_params.bess_power_capex_per_mw
    bess_energy_cost = bess_energy_mwh * cost_params.bess_energy_capex_per_mwh
    
    total_cost = solar_cost + wind_cost + bess_power_cost + bess_energy_cost
    return total_cost


def calculate_annual_opex(solar_ac_mw: float, wind_mw: float, 
                         bess_energy_mwh: float, 
                         cost_params: CostParameters) -> float:
    """
    Calculate annual operating expenses.
    
    Returns:
        Annual OPEX in Crore INR
    """
    solar_opex = solar_ac_mw * cost_params.solar_opex_per_mw_year
    wind_opex = wind_mw * cost_params.wind_opex_per_mw_year
    bess_opex = bess_energy_mwh * cost_params.bess_opex_per_mwh_year
    
    total_opex = solar_opex + wind_opex + bess_opex + cost_params.fixed_opex
    return total_opex


def calculate_depreciation_schedule(project_cost: float, 
                                    fin_params: FinancialParameters) -> np.ndarray:
    """
    Calculate depreciation for each year (straight-line method).
    Excel uses: 4.67% for years 1-15, 3% for years 16-25
    
    Returns:
        Array of depreciation amounts for each year (Crore INR)
    """
    n_years = fin_params.project_lifetime_years
    dep_schedule = np.zeros(n_years)
    
    for year in range(1, n_years + 1):
        if year <= fin_params.depreciation_switchover_year:
            dep_schedule[year - 1] = project_cost * fin_params.depreciation_rate_early
        else:
            dep_schedule[year - 1] = project_cost * fin_params.depreciation_rate_late
    
    return dep_schedule


def calculate_loan_schedule(project_cost: float, 
                           fin_params: FinancialParameters) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate loan repayment schedule.
    
    Returns:
        Tuple of (principal_payments, interest_payments, outstanding_balance) arrays
    """
    n_years = fin_params.project_lifetime_years
    loan_amount = project_cost * fin_params.loan_fraction
    
    principal_payments = np.zeros(n_years)
    interest_payments = np.zeros(n_years)
    outstanding = np.zeros(n_years)
    
    # Simple loan: equal principal payments over loan term
    annual_principal = loan_amount / fin_params.loan_term_years if fin_params.loan_term_years > 0 else 0
    
    outstanding_balance = loan_amount
    for year in range(n_years):
        outstanding[year] = outstanding_balance
        
        if year < fin_params.loan_term_years:
            principal_payments[year] = annual_principal
            interest_payments[year] = outstanding_balance * fin_params.interest_rate
            outstanding_balance -= annual_principal
        else:
            principal_payments[year] = 0
            interest_payments[year] = 0
    
    return principal_payments, interest_payments, outstanding


def calculate_lcoe(project_cost: float, 
                  annual_opex: float,
                  annual_energy_kwh: np.ndarray,  # Array of energy for each year
                  fin_params: FinancialParameters,
                  cost_params: CostParameters) -> dict:
    """
    Calculate LCOE using the full formula.
    
    Args:
        project_cost: Total capital cost (Crore INR)
        annual_opex: Annual operating expenses (Crore INR)
        annual_energy_kwh: Array of delivered energy for each year (kWh)
        fin_params: Financial parameters
        cost_params: Cost parameters
        
    Returns:
        Dictionary with LCOE and breakdown components
    """
    n_years = fin_params.project_lifetime_years
    dr = fin_params.discount_rate
    tr = fin_params.tax_rate
    
    # Calculate schedules
    depreciation = calculate_depreciation_schedule(project_cost, fin_params)
    principal, interest, outstanding = calculate_loan_schedule(project_cost, fin_params)
    loan_payments = principal + interest
    
    # Residual value at end of project
    residual_value = project_cost * cost_params.residual_value_fraction
    
    # Calculate NPV components
    pci_equity = project_cost * fin_params.equity_fraction
    
    # Tax shield from depreciation and interest: (DEP + INT) × TR / (1+DR)^n
    tax_shield_npv = 0.0
    for year in range(n_years):
        n = year + 1
        tax_shield = (depreciation[year] + interest[year]) * tr / ((1 + dr) ** n)
        tax_shield_npv += tax_shield
    
    # Loan payments: LP / (1+DR)^n
    loan_payment_npv = 0.0
    for year in range(n_years):
        n = year + 1
        loan_payment_npv += loan_payments[year] / ((1 + dr) ** n)
    
    # Operating costs with escalation (tax applied in final formula): AO × (1+esc)^year / (1+DR)^n
    opex_npv = 0.0
    for year in range(n_years):
        n = year + 1
        escalated_opex = annual_opex * ((1 + fin_params.opex_escalation) ** year)
        opex_npv += escalated_opex / ((1 + dr) ** n)
    
    # Residual value: RV / (1+DR)^n
    residual_npv = residual_value / ((1 + dr) ** n_years)
    
    # Total cost NPV (apply tax to OPEX in final formula, matching Excel)
    total_cost_npv = pci_equity - tax_shield_npv + loan_payment_npv + opex_npv * (1 - tr) - residual_npv
    
    # Energy NPV: Σ kWh × (1-SDR)^n / (1+DR)^n
    energy_npv = 0.0
    for year in range(n_years):
        n = year + 1
        degraded_energy = annual_energy_kwh[year] * ((1 - fin_params.system_degradation) ** (n - 1))
        energy_npv += degraded_energy / ((1 + dr) ** n)
    
    # LCOE in INR per kWh
    lcoe = total_cost_npv / energy_npv if energy_npv > 0 else float('inf')
    
    # Convert to more readable units (Crore to actual rupees: 1 Crore = 10^7)
    # Cost is in Crore, energy in kWh
    # LCOE = (Crore × 10^7) / kWh = (10^7 / kWh) = INR/kWh × 10^7 / 10^7 = INR/kWh
    # Actually need to convert: Crore / kWh = 10^7 INR / kWh → need to keep units consistent
    
    return {
        'lcoe_inr_per_kwh': lcoe * 1e7,  # Convert Crore/kWh to INR/kWh
        'lcoe_crore_per_kwh': lcoe,
        'total_cost_npv_crore': total_cost_npv,
        'energy_npv_kwh': energy_npv,
        'project_cost_crore': project_cost,
        'equity_crore': pci_equity,
        'loan_amount_crore': project_cost * fin_params.loan_fraction,
        'annual_opex_crore': annual_opex,
        'tax_shield_npv_crore': tax_shield_npv,
        'loan_payment_npv_crore': loan_payment_npv,
        'opex_npv_crore': opex_npv,
        'residual_npv_crore': residual_npv,
        'first_year_energy_kwh': annual_energy_kwh[0] if len(annual_energy_kwh) > 0 else 0,
        'lifetime_energy_kwh': annual_energy_kwh.sum(),
    }


def format_lcoe_results(results: dict) -> str:
    """Format LCOE results for display."""
    lines = [
        "="*70,
        "LCOE CALCULATION RESULTS",
        "="*70,
        "",
        f"LCOE: ₹{results['lcoe_inr_per_kwh']:.4f} per kWh",
        "",
        "Capital Structure:",
        f"  Project Cost:     ₹{results['project_cost_crore']:.2f} Crore",
        f"  Equity (30%):     ₹{results['equity_crore']:.2f} Crore",
        f"  Loan (70%):       ₹{results['loan_amount_crore']:.2f} Crore",
        "",
        "Operating Costs:",
        f"  Annual OPEX:      ₹{results['annual_opex_crore']:.2f} Crore/year",
        "",
        "NPV Components (₹ Crore):",
        f"  Total Cost NPV:   ₹{results['total_cost_npv_crore']:.2f}",
        f"    - Tax Shield:   ₹{results['tax_shield_npv_crore']:.2f}",
        f"    - Loan Payment: ₹{results['loan_payment_npv_crore']:.2f}",
        f"    - OPEX:         ₹{results['opex_npv_crore']:.2f}",
        f"    - Residual:     ₹{results['residual_npv_crore']:.2f}",
        "",
        "Energy:",
        f"  Year 1:           {results['first_year_energy_kwh']/1e9:.2f} GWh",
        f"  Lifetime:         {results['lifetime_energy_kwh']/1e9:.2f} GWh",
        f"  NPV:              {results['energy_npv_kwh']/1e9:.2f} GWh",
        "="*70,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Example calculation
    cost_params = CostParameters()
    fin_params = FinancialParameters()
    
    # Example project: 830 MW solar, 0 MW wind, 50 MW / 200 MWh BESS
    solar_mw = 830
    wind_mw = 0
    bess_power_mw = 50
    bess_energy_mwh = 200
    
    project_cost = calculate_project_cost(solar_mw, wind_mw, bess_power_mw, bess_energy_mwh, cost_params)
    annual_opex = calculate_annual_opex(solar_mw, wind_mw, bess_energy_mwh, cost_params)
    
    # Example: assume 1.5 TWh first year energy delivered
    first_year_energy = 1.5e12  # 1.5 TWh in kWh
    annual_energy = np.full(fin_params.project_lifetime_years, first_year_energy)
    
    results = calculate_lcoe(project_cost, annual_opex, annual_energy, fin_params, cost_params)
    print(format_lcoe_results(results))
