#!/usr/bin/env python3
"""
Command-line interface for Hybrid RE System Simulation.
Usage: python cli.py simulate --config config.json
       python cli.py optimize --config config.json --output results.csv
"""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd

from hourly_sim_skeleton import Config, simulate_hourly, assign_tod, load_profiles_from_workbook
from lcoe_calculator import calculate_lcoe
from optimizer import optimize_system

def cmd_simulate(args):
    """Run a single simulation"""
    print(f"Loading configuration from {args.config}")
    
    # Load config
    with open(args.config) as f:
        config_data = json.load(f)
    
    # Create Config object
    from dataclasses import fields
    valid_fields = {f.name for f in fields(Config)}
    filtered_config = {k: v for k, v in config_data.items() if k in valid_fields}
    cfg = Config(**filtered_config)
    
    print(f"Configuration: Solar={cfg.solar_ac_mw}MW, Wind={cfg.wind_wtg_count}WTG, BESS={cfg.bess_power_mw}MW/{cfg.bess_energy_mwh}MWh")
    
    # Load profiles
    data_dir = Path(args.data_dir)
    df_gen = pd.read_csv(data_dir / "generation_profiles.csv")
    df_load = pd.read_csv(data_dir / "load_profile.csv")
    
    dt = pd.date_range('2024-01-01', periods=8760, freq='h')
    hour = df_gen['hour_of_day'].values if 'hour_of_day' in df_gen.columns else dt.hour.to_numpy()
    
    df_profiles = pd.DataFrame({
        'dt': dt,
        'month': dt.month.to_numpy(),
        'day': dt.day.to_numpy(),
        'hour': hour,
        'tod': assign_tod(hour),
        'solar_mwh_per_mw': df_gen['solar_kwh_per_mw'].values / 1000.0,
        'wind_mwh_per_wtg': df_gen['wind_kwh_per_wtg'].values / 1000.0,
        'load_mw': df_load['load_mw'].values,
    })
    
    print("Running simulation...")
    df_sim = simulate_hourly(cfg, df_profiles)
    
    # Calculate LCOE
    financial_file = data_dir / "financial_params.json"
    if financial_file.exists():
        try:
            lcoe = calculate_lcoe(cfg, df_sim, str(financial_file))
            print(f"\nLCOE: ₹{lcoe:.4f}/kWh")
        except Exception as e:
            print(f"Warning: Could not calculate LCOE: {e}")
            lcoe = None
    else:
        lcoe = None
    
    # Print results
    print("\n=== Simulation Results ===")
    print(f"Solar Generation:     {df_sim['solar_gen'].sum():>12,.0f} MWh")
    print(f"Wind Generation:      {df_sim['wind_gen'].sum():>12,.0f} MWh")
    print(f"Total Generation:     {df_sim['gen_total'].sum():>12,.0f} MWh")
    print(f"BESS Charge:          {df_sim['bess_charge_inj'].sum():>12,.0f} MWh")
    print(f"BESS Discharge:       {df_sim['bess_discharge_inj'].sum():>12,.0f} MWh")
    print(f"Curtailment:          {df_sim['curtailed'].sum():>12,.0f} MWh")
    print(f"Grid Import:          {df_sim['from_grid'].sum():>12,.0f} MWh")
    print(f"Direct Delivered:     {df_sim['direct_delivered'].sum():>12,.0f} MWh")
    print(f"From BESS Delivered:  {df_sim['from_bess_delivered'].sum():>12,.0f} MWh")
    print(f"Total Delivered:      {df_sim['delivered_total'].sum():>12,.0f} MWh")
    
    # Save detailed output if requested
    if args.output:
        df_sim.to_csv(args.output, index=False)
        print(f"\nDetailed results saved to {args.output}")
        
        # Also save summary
        if args.summary:
            summary = {
                'config': config_data,
                'lcoe': lcoe,
                'results': {
                    'solar_gen_mwh': float(df_sim['solar_gen'].sum()),
                    'wind_gen_mwh': float(df_sim['wind_gen'].sum()),
                    'total_gen_mwh': float(df_sim['gen_total'].sum()),
                    'bess_charge_mwh': float(df_sim['bess_charge_inj'].sum()),
                    'bess_discharge_mwh': float(df_sim['bess_discharge_inj'].sum()),
                    'curtailment_mwh': float(df_sim['curtailed'].sum()),
                    'grid_import_mwh': float(df_sim['from_grid'].sum()),
                    'delivered_total_mwh': float(df_sim['delivered_total'].sum()),
                }
            }
            with open(args.summary, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Summary saved to {args.summary}")
    
    return 0

def cmd_optimize(args):
    """Run optimization"""
    print(f"Loading base configuration from {args.config}")
    
    with open(args.config) as f:
        config_data = json.load(f)
    
    print(f"\nOptimization settings:")
    print(f"  Solar: {args.solar_min}-{args.solar_max} MW (step {args.solar_step})")
    print(f"  Wind: {args.wind_min}-{args.wind_max} WTG (step {args.wind_step})")
    if args.bess_power_min is not None:
        print(f"  BESS Power: {args.bess_power_min}-{args.bess_power_max} MW")
        print(f"  BESS Energy: {args.bess_energy_min}-{args.bess_energy_max} MWh")
    
    # Call optimizer
    try:
        from optimizer import optimize_system
        
        results = optimize_system(
            base_config=config_data,
            data_dir=args.data_dir,
            solar_range=(args.solar_min, args.solar_max, args.solar_step),
            wind_range=(args.wind_min, args.wind_max, args.wind_step),
            bess_power_range=(args.bess_power_min, args.bess_power_max) if args.bess_power_min is not None else None,
            bess_energy_range=(args.bess_energy_min, args.bess_energy_max) if args.bess_energy_min is not None else None,
            output_file=args.output,
            max_configs=args.max_configs
        )
        
        print(f"\n✅ Optimization complete! Results saved to {args.output}")
        
    except ImportError:
        print("Error: optimizer.py not available or needs updates for standalone mode")
        return 1
    
    return 0

def cmd_extract(args):
    """Extract standalone data from Excel"""
    from utils.extract_standalone_data import extract_standalone_data
    
    print(f"Extracting data from {args.excel_file}")
    extract_standalone_data(args.excel_file, args.output_dir)
    print(f"✅ Data extracted to {args.output_dir}/")
    
    return 0

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid RE System Simulator & Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract data from Excel
  python cli.py extract oa_hybrid.xlsx --output-dir standalone_data
  
  # Run simulation
  python cli.py simulate --config standalone_data/system_config.json
  
  # Run simulation with output
  python cli.py simulate --config my_config.json --output results.csv --summary summary.json
  
  # Run optimization
  python cli.py optimize --config standalone_data/system_config.json \\
      --solar-min 500 --solar-max 1000 --solar-step 50 \\
      --wind-min 0 --wind-max 20 --wind-step 5 \\
      --output optimization_results.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Simulate command
    sim_parser = subparsers.add_parser('simulate', help='Run a single simulation')
    sim_parser.add_argument('--config', required=True, help='Configuration JSON file')
    sim_parser.add_argument('--data-dir', default='standalone_data', help='Directory with profile data')
    sim_parser.add_argument('--output', help='Output CSV file for detailed results')
    sim_parser.add_argument('--summary', help='Output JSON file for summary')
    
    # Optimize command
    opt_parser = subparsers.add_parser('optimize', help='Run optimization')
    opt_parser.add_argument('--config', required=True, help='Base configuration JSON file')
    opt_parser.add_argument('--data-dir', default='standalone_data', help='Directory with profile data')
    opt_parser.add_argument('--output', required=True, help='Output CSV file for results')
    opt_parser.add_argument('--solar-min', type=float, default=0, help='Min solar capacity (MW)')
    opt_parser.add_argument('--solar-max', type=float, default=1000, help='Max solar capacity (MW)')
    opt_parser.add_argument('--solar-step', type=float, default=50, help='Solar step size (MW)')
    opt_parser.add_argument('--wind-min', type=int, default=0, help='Min wind WTG count')
    opt_parser.add_argument('--wind-max', type=int, default=50, help='Max wind WTG count')
    opt_parser.add_argument('--wind-step', type=int, default=5, help='Wind step size')
    opt_parser.add_argument('--bess-power-min', type=float, help='Min BESS power (MW)')
    opt_parser.add_argument('--bess-power-max', type=float, help='Max BESS power (MW)')
    opt_parser.add_argument('--bess-energy-min', type=float, help='Min BESS energy (MWh)')
    opt_parser.add_argument('--bess-energy-max', type=float, help='Max BESS energy (MWh)')
    opt_parser.add_argument('--max-configs', type=int, default=1000, help='Max configurations to evaluate')
    
    # Extract command
    ext_parser = subparsers.add_parser('extract', help='Extract data from Excel to standalone files')
    ext_parser.add_argument('excel_file', help='Input Excel file')
    ext_parser.add_argument('--output-dir', default='standalone_data', help='Output directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == 'simulate':
        return cmd_simulate(args)
    elif args.command == 'optimize':
        return cmd_optimize(args)
    elif args.command == 'extract':
        return cmd_extract(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
