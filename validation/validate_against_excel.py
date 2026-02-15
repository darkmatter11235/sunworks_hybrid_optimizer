#!/usr/bin/env python3
"""
Validation utility to compare Python simulation results against Excel workbook values.

Usage:
  python validate_against_excel.py oa_hybrid.xlsx

This loads the workbook, runs the Python simulation with the same inputs,
and compares key columns to identify any discrepancies.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import openpyxl
from hourly_sim_skeleton import Config, load_profiles_from_workbook, simulate_hourly
from banking_settlement_skeleton import SettlementConfig, settle

# Column mapping: Excel column letter -> Python dataframe column name
HOURLY_COLUMNS = {
    "V": "to_consumption_inj",
    "X": "to_battery_inj",
    "Y": "to_banking_inj",
    "Z": "curtailed",
    "AG": "net_injection",
    "AJ": None,  # Component of discharge (not directly in df)
    "AK": None,  # Component of discharge (not directly in df)
    "AL": "bess_discharge_inj",
    "AM": "bess_charge_inj",
    "AN": "soc_mwh",
    "AZ": "banked_after_losses",
    "BA": "delivered_total",
    "BE": "from_grid",
}

def load_excel_column(xlsx_path: str, sheet_name: str, col_letter: str, start_row: int = 9, n_rows: int = 8760) -> np.ndarray:
    """
    Load a column from Excel workbook (data_only=True for computed values).
    """
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb[sheet_name]
    col_num = openpyxl.utils.column_index_from_string(col_letter)
    values = np.array([ws.cell(start_row + i - 1, col_num).value for i in range(1, n_rows + 1)], dtype=float)
    return values

def compare_column(py_col: np.ndarray, excel_col: np.ndarray, col_name: str, tolerance: float = 1e-3):
    """
    Compare Python and Excel columns and report differences.
    """
    diff = py_col - excel_col
    abs_diff = np.abs(diff)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    sum_py = np.sum(py_col)
    sum_excel = np.sum(excel_col)
    
    match = max_diff < tolerance
    status = "✓ MATCH" if match else "✗ MISMATCH"
    
    print(f"\n{col_name}:")
    print(f"  Status: {status}")
    print(f"  Python sum:  {sum_py:,.2f}")
    print(f"  Excel sum:   {sum_excel:,.2f}")
    print(f"  Diff sum:    {sum_py - sum_excel:,.2f}")
    print(f"  Max diff:    {max_diff:.6f}")
    print(f"  Mean diff:   {mean_diff:.6f}")
    
    if not match:
        # Show first few mismatches
        mismatches = np.where(abs_diff > tolerance)[0]
        print(f"  Mismatches:  {len(mismatches)} hours")
        if len(mismatches) > 0:
            print(f"  First 5 mismatch indices: {mismatches[:5]}")
            for idx in mismatches[:5]:
                print(f"    Hour {idx}: Python={py_col[idx]:.4f}, Excel={excel_col[idx]:.4f}, Diff={diff[idx]:.4f}")
    
    return match

def validate_hourly(xlsx_path: str, tolerance: float = 1e-3):
    """
    Run the Python simulation and compare against Excel Hourly Data sheet.
    """
    print(f"Loading workbook: {xlsx_path}")
    
    # Load profiles and run simulation
    cfg = Config()
    df_profiles = load_profiles_from_workbook(xlsx_path, cfg)
    df_sim = simulate_hourly(cfg, df_profiles)
    
    print("\n" + "="*70)
    print("HOURLY SIMULATION VALIDATION")
    print("="*70)
    
    all_match = True
    for excel_col, py_col in HOURLY_COLUMNS.items():
        if py_col is None:
            continue
        
        try:
            excel_values = load_excel_column(xlsx_path, " Hourly Data", excel_col)
            py_values = df_sim[py_col].to_numpy()
            
            match = compare_column(py_values, excel_values, f"Col {excel_col} ({py_col})", tolerance)
            all_match = all_match and match
        except Exception as e:
            print(f"\n{excel_col} ({py_col}): ERROR - {e}")
            all_match = False
    
    print("\n" + "="*70)
    if all_match:
        print("✓ ALL COLUMNS MATCH!")
    else:
        print("✗ SOME COLUMNS HAVE MISMATCHES")
    print("="*70)
    
    return df_sim, all_match

def validate_settlement(df_hourly: pd.DataFrame, xlsx_path: str):
    """
    Run banking settlement and compare key metrics.
    """
    print("\n" + "="*70)
    print("SETTLEMENT VALIDATION")
    print("="*70)
    
    scfg = SettlementConfig()
    results = settle(df_hourly, scfg)
    
    print("\nBanking totals by ToD:")
    print(results["banked"].sum())
    print("\nBanked after charge:")
    print(results["banked_after_charge"].sum())
    print("\nTotal used:")
    print(results["used_total"].sum())
    print("\nUnutilized:")
    print(results["unutilized"].sum())
    
    # TODO: Add comparison with Excel Banking Settlement sheet if needed
    return results

def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_against_excel.py <xlsx_path>")
        raise SystemExit(2)
    
    xlsx_path = sys.argv[1]
    tolerance = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
    
    df_sim, hourly_match = validate_hourly(xlsx_path, tolerance)
    results = validate_settlement(df_sim, xlsx_path)
    
    if hourly_match:
        print("\n✓ Validation PASSED")
        return 0
    else:
        print("\n✗ Validation FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
