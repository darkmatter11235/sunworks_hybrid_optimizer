#!/usr/bin/env python3
"""
Optimization framework to find the best solar/wind/BESS configuration.

Searches over user-specified ranges to minimize LCOE while meeting constraints.
"""
from __future__ import annotations

import sys
import argparse
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import itertools
import pandas as pd
import numpy as np
from pathlib import Path

from hourly_sim_skeleton import Config, load_profiles_from_workbook, simulate_hourly
from banking_settlement_skeleton import SettlementConfig, settle
from lcoe_calculator import (
    CostParameters, FinancialParameters,
    calculate_project_cost, calculate_annual_opex, calculate_lcoe,
    format_lcoe_results
)


@dataclass
class OptimizationConfig:
    """Configuration for optimization search space."""
    # Solar ranges
    solar_ac_mw_min: float = 600
    solar_ac_mw_max: float = 1200
    solar_ac_mw_step: float = 100
    
    # Wind ranges
    wind_wtg_count_min: int = 0
    wind_wtg_count_max: int = 100
    wind_wtg_count_step: int = 20
    wind_wtg_capacity_mw: float = 3.3  # MW per turbine
    
    # BESS Power ranges
    bess_power_mw_min: float = 0
    bess_power_mw_max: float = 200
    bess_power_mw_step: float = 50
    
    # BESS Energy ranges (MWh)
    bess_energy_mwh_min: float = 0
    bess_energy_mwh_max: float = 800
    bess_energy_mwh_step: float = 200
    
    # Constraints
    max_evacuation_limit_mw: float = 1200
    min_captive_percent: float = 0.0  # Minimum % of load from captive
    max_grid_import_percent: float = 100.0  # Maximum % of load from grid
    
    # BESS mode
    bess_mode: str = "PV_SHIFT"  # or "FDRE"


@dataclass
class OptimizationResult:
    """Result of a single configuration evaluation."""
    solar_ac_mw: float
    wind_wtg_count: int
    wind_mw: float
    bess_power_mw: float
    bess_energy_mwh: float
    
    lcoe_inr_per_kwh: float
    project_cost_crore: float
    annual_opex_crore: float
    
    annual_generation_gwh: float
    annual_load_gwh: float
    captive_supply_gwh: float
    grid_import_gwh: float
    curtailment_gwh: float
    
    captive_percent: float
    grid_import_percent: float
    cuf_percent: float
    
    is_feasible: bool
    constraint_violations: List[str]


def generate_configurations(opt_config: OptimizationConfig) -> List[Tuple[float, int, float, float]]:
    """
    Generate all combinations of (solar_mw, wind_wtg, bess_power_mw, bess_energy_mwh).
    
    Returns:
        List of configuration tuples
    """
    solar_range = np.arange(
        opt_config.solar_ac_mw_min,
        opt_config.solar_ac_mw_max + opt_config.solar_ac_mw_step/2,
        opt_config.solar_ac_mw_step
    )
    
    wind_range = np.arange(
        opt_config.wind_wtg_count_min,
        opt_config.wind_wtg_count_max + opt_config.wind_wtg_count_step/2,
        opt_config.wind_wtg_count_step
    )
    
    bess_power_range = np.arange(
        opt_config.bess_power_mw_min,
        opt_config.bess_power_mw_max + opt_config.bess_power_mw_step/2,
        opt_config.bess_power_mw_step
    )
    
    bess_energy_range = np.arange(
        opt_config.bess_energy_mwh_min,
        opt_config.bess_energy_mwh_max + opt_config.bess_energy_mwh_step/2,
        opt_config.bess_energy_mwh_step
    )
    
    # Generate all combinations
    configurations = list(itertools.product(
        solar_range, wind_range, bess_power_range, bess_energy_range
    ))
    
    return [(float(s), int(w), float(bp), float(be)) for s, w, bp, be in configurations]


def evaluate_configuration(
    solar_ac_mw: float,
    wind_wtg_count: int,
    bess_power_mw: float,
    bess_energy_mwh: float,
    xlsx_path: str,
    opt_config: OptimizationConfig,
    cost_params: CostParameters,
    fin_params: FinancialParameters,
    base_config: Config
) -> OptimizationResult:
    """
    Evaluate a single configuration: run simulation and calculate LCOE.
    """
    # Update configuration
    cfg = Config(
        year=base_config.year,
        total_load_mw=base_config.total_load_mw,
        existing_solar_mwp=base_config.existing_solar_mwp,
        contracted_demand_mw=base_config.contracted_demand_mw,
        evac_limit_mw=opt_config.max_evacuation_limit_mw,
        tl_loss=base_config.tl_loss,
        wheeling_loss=base_config.wheeling_loss,
        dc_ac_ratio=base_config.dc_ac_ratio,
        solar_ac_mw=solar_ac_mw,
        wind_wtg_count=wind_wtg_count,
        wind_expected_cuf=base_config.wind_expected_cuf,
        wind_reference_cuf=base_config.wind_reference_cuf,
        solar_degrad=base_config.solar_degrad,
        wind_degrad=base_config.wind_degrad,
        p_multiplier=base_config.p_multiplier,
        banking_enabled=base_config.banking_enabled,
        bess_mode=opt_config.bess_mode,
        one_way_eff=base_config.one_way_eff,
        bess_power_mw=bess_power_mw,
        bess_energy_mwh=bess_energy_mwh,
        soc_start_gwh=base_config.soc_start_gwh,
    )
    
    # Load profiles (cached or reuse if already loaded)
    df_profiles = load_profiles_from_workbook(xlsx_path, cfg)
    
    # Run hourly simulation
    df_hourly = simulate_hourly(cfg, df_profiles)
    
    # Run settlement
    scfg = SettlementConfig()
    settlement_results = settle(df_hourly, scfg)
    
    # Calculate metrics
    annual_generation = df_hourly["gen_total"].sum() / 1e6  # GWh
    annual_load = df_hourly["net_load"].sum() / 1e6  # GWh
    captive_supply = (df_hourly["delivered_total"].sum() + settlement_results["annual_summary"]["total_banked_used"]) / 1e6  # GWh
    grid_import = df_hourly["from_grid"].sum() / 1e6  # GWh
    curtailment = df_hourly["curtailed"].sum() / 1e6  # GWh
    
    captive_percent = (captive_supply / annual_load * 100) if annual_load > 0 else 0
    grid_import_percent = (grid_import / annual_load * 100) if annual_load > 0 else 0
    
    wind_mw = wind_wtg_count * opt_config.wind_wtg_capacity_mw
    total_capacity = solar_ac_mw + wind_mw
    cuf_percent = (annual_generation * 1e6 / (total_capacity * 8760)) * 100 if total_capacity > 0 else 0
    
    # Check constraints
    violations = []
    is_feasible = True
    
    if captive_percent < opt_config.min_captive_percent:
        violations.append(f"Captive {captive_percent:.1f}% < {opt_config.min_captive_percent:.1f}%")
        is_feasible = False
    
    if grid_import_percent > opt_config.max_grid_import_percent:
        violations.append(f"Grid import {grid_import_percent:.1f}% > {opt_config.max_grid_import_percent:.1f}%")
        is_feasible = False
    
    # Calculate costs
    project_cost = calculate_project_cost(solar_ac_mw, wind_mw, bess_power_mw, bess_energy_mwh, cost_params)
    annual_opex = calculate_annual_opex(solar_ac_mw, wind_mw, bess_energy_mwh, cost_params)
    
    # Energy delivered over project lifetime (with degradation)
    first_year_energy_kwh = captive_supply * 1e9  # GWh to kWh
    annual_energy_array = np.full(fin_params.project_lifetime_years, first_year_energy_kwh)
    
    # Calculate LCOE
    lcoe_results = calculate_lcoe(project_cost, annual_opex, annual_energy_array, fin_params, cost_params)
    
    return OptimizationResult(
        solar_ac_mw=solar_ac_mw,
        wind_wtg_count=wind_wtg_count,
        wind_mw=wind_mw,
        bess_power_mw=bess_power_mw,
        bess_energy_mwh=bess_energy_mwh,
        lcoe_inr_per_kwh=lcoe_results['lcoe_inr_per_kwh'],
        project_cost_crore=project_cost,
        annual_opex_crore=annual_opex,
        annual_generation_gwh=annual_generation,
        annual_load_gwh=annual_load,
        captive_supply_gwh=captive_supply,
        grid_import_gwh=grid_import,
        curtailment_gwh=curtailment,
        captive_percent=captive_percent,
        grid_import_percent=grid_import_percent,
        cuf_percent=cuf_percent,
        is_feasible=is_feasible,
        constraint_violations=violations,
    )


def optimize(
    xlsx_path: str,
    opt_config: OptimizationConfig,
    cost_params: CostParameters,
    fin_params: FinancialParameters,
    base_config: Config,
    output_path: Optional[str] = None
) -> Tuple[OptimizationResult, pd.DataFrame]:
    """
    Run optimization over all configurations.
    
    Returns:
        Tuple of (best_result, all_results_dataframe)
    """
    print(f"\nGenerating configurations...")
    configurations = generate_configurations(opt_config)
    print(f"Total configurations to evaluate: {len(configurations)}")
    
    results = []
    for i, (solar, wind, bess_p, bess_e) in enumerate(configurations):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Evaluating configuration {i+1}/{len(configurations)}: "
                  f"Solar={solar}MW, Wind={wind}×3.3MW, BESS={bess_p}MW/{bess_e}MWh")
        
        try:
            result = evaluate_configuration(
                solar, wind, bess_p, bess_e,
                xlsx_path, opt_config, cost_params, fin_params, base_config
            )
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Convert to DataFrame
    df_results = pd.DataFrame([asdict(r) for r in results])
    
    # Find best feasible solution
    feasible_results = df_results[df_results['is_feasible']]
    
    if len(feasible_results) == 0:
        print("\nWARNING: No feasible solutions found!")
        best_idx = df_results['lcoe_inr_per_kwh'].idxmin()
    else:
        best_idx = feasible_results['lcoe_inr_per_kwh'].idxmin()
    
    best_result_dict = df_results.loc[best_idx].to_dict()
    best_result = OptimizationResult(**best_result_dict)
    
    # Save results if output path specified
    if output_path:
        df_results.to_csv(output_path, index=False)
        print(f"\nAll results saved to: {output_path}")
    
    return best_result, df_results


def print_optimization_summary(best_result: OptimizationResult, df_results: pd.DataFrame):
    """Print summary of optimization results."""
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    
    print(f"\nBest Configuration:")
    print(f"  Solar AC:         {best_result.solar_ac_mw:.0f} MW")
    print(f"  Wind:             {best_result.wind_wtg_count} × 3.3 MW = {best_result.wind_mw:.1f} MW")
    print(f"  BESS:             {best_result.bess_power_mw:.0f} MW / {best_result.bess_energy_mwh:.0f} MWh")
    print(f"  Total Capacity:   {best_result.solar_ac_mw + best_result.wind_mw:.1f} MW")
    
    print(f"\nEconomics:")
    print(f"  LCOE:             ₹{best_result.lcoe_inr_per_kwh:.4f} /kWh")
    print(f"  Project Cost:     ₹{best_result.project_cost_crore:.2f} Crore")
    print(f"  Annual OPEX:      ₹{best_result.annual_opex_crore:.2f} Crore")
    
    print(f"\nPerformance:")
    print(f"  Generation:       {best_result.annual_generation_gwh:.2f} GWh/year")
    print(f"  Load:             {best_result.annual_load_gwh:.2f} GWh/year")
    print(f"  Captive Supply:   {best_result.captive_supply_gwh:.2f} GWh ({best_result.captive_percent:.1f}%)")
    print(f"  Grid Import:      {best_result.grid_import_gwh:.2f} GWh ({best_result.grid_import_percent:.1f}%)")
    print(f"  Curtailment:      {best_result.curtailment_gwh:.2f} GWh")
    print(f"  CUF:              {best_result.cuf_percent:.1f}%")
    
    print(f"\nSolution Status:  {'✓ FEASIBLE' if best_result.is_feasible else '✗ INFEASIBLE'}")
    if best_result.constraint_violations:
        print(f"  Violations: {', '.join(best_result.constraint_violations)}")
    
    # Top 5 results
    print(f"\nTop 5 Configurations by LCOE:")
    top_5 = df_results.nsmallest(5, 'lcoe_inr_per_kwh')[['solar_ac_mw', 'wind_wtg_count', 'bess_power_mw', 'bess_energy_mwh', 'lcoe_inr_per_kwh', 'captive_percent', 'is_feasible']]
    print(top_5.to_string(index=False))
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Optimize solar/wind/BESS configuration for minimum LCOE")
    parser.add_argument("workbook", help="Path to Excel workbook")
    parser.add_argument("-o", "--output", help="Output CSV file for all results", default="optimization_results.csv")
    parser.add_argument("--config", help="JSON config file for optimization parameters", default=None)
    args = parser.parse_args()
    
    # Load or use default configuration
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            opt_config_dict = json.load(f)
        opt_config = OptimizationConfig(**opt_config_dict)
    else:
        opt_config = OptimizationConfig()
    
    cost_params = CostParameters()
    fin_params = FinancialParameters()
    
    # Base config from workbook (will be modified for each configuration)
    base_config = Config()
    
    # Run optimization
    best_result, df_results = optimize(
        args.workbook,
        opt_config,
        cost_params,
        fin_params,
        base_config,
        args.output
    )
    
    # Print summary
    print_optimization_summary(best_result, df_results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
