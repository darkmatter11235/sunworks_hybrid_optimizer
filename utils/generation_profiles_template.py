#!/usr/bin/env python3
"""Create and convert a multi-ratio generation profile Excel template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

SOLAR_PREFIX = "solar_kwh_per_mw_dcac_"
HOURS_PER_YEAR = 8760


def _normalize_ratio(ratio: float) -> str:
    return f"{ratio:.6f}".rstrip("0").rstrip(".")


def _ratio_filename_token(ratio: float) -> str:
    return _normalize_ratio(ratio).replace(".", "_")


def _build_template_dataframe(dc_ac_ratios: Iterable[float]) -> pd.DataFrame:
    hour_index = np.arange(HOURS_PER_YEAR)
    df = pd.DataFrame(
        {
            "hour_index": hour_index,
            "hour_of_day": hour_index % 24,
            "wind_kwh_per_wtg": np.zeros(HOURS_PER_YEAR),
        }
    )
    for ratio in sorted(dc_ac_ratios):
        df[f"{SOLAR_PREFIX}{_normalize_ratio(ratio)}"] = np.zeros(HOURS_PER_YEAR)
    return df


def create_template(output_path: Path, dc_ac_ratios: List[float]) -> Path:
    if not dc_ac_ratios:
        raise ValueError("At least one dc_ac_ratio is required.")

    wb = Workbook()
    ws_meta = wb.active
    ws_meta.title = "meta"
    ws_instructions = wb.create_sheet("instructions")
    ws_plant = wb.create_sheet("plant_info")
    ws_hourly = wb.create_sheet("hourly_profiles")

    ws_meta.append(["field", "value"])
    ws_meta.append(["description", "Generation profile input template for simulator"])
    ws_meta.append(["hours_expected", HOURS_PER_YEAR])
    ws_meta.append(["required_base_columns", "hour_index,hour_of_day,wind_kwh_per_wtg"])
    ws_meta.append(["solar_column_prefix", SOLAR_PREFIX])
    ws_meta.append(["example_solar_column", f"{SOLAR_PREFIX}1.45"])
    ws_meta.append(["notes", "Fill hourly_profiles with 8760 rows. Keep header names unchanged."])

    ws_instructions.append(["step", "instruction"])
    ws_instructions.append([1, "Go to plant_info and fill place_name, latitude, longitude (and optional plant_name)."])
    ws_instructions.append([2, "Go to hourly_profiles and fill exactly 8760 rows (one row per hour)."])
    ws_instructions.append([3, "Do not rename required columns: hour_index, hour_of_day, wind_kwh_per_wtg."])
    ws_instructions.append([4, f"For each dc_ac_ratio, add/use a solar column named {SOLAR_PREFIX}<ratio>."])
    ws_instructions.append([5, "Use dot format for ratio names (example: solar_kwh_per_mw_dcac_1.45)."])
    ws_instructions.append([6, "solar_kwh_per_mw and wind_kwh_per_wtg values must be kWh basis."])
    ws_instructions.append([7, "hour_index should run 0..8759 and hour_of_day should be 0..23 repeating."])
    ws_instructions.append([8, "After filling, run convert command to generate generation_profiles CSVs."])

    ws_plant.append(["field", "value", "required", "notes"])
    ws_plant.append(["plant_name", "", "no", "Plant/project label for your reference"])
    ws_plant.append(["place_name", "", "yes", "Village/Town/City or site name"])
    ws_plant.append(["latitude", "", "yes", "Decimal degrees, range -90 to 90"])
    ws_plant.append(["longitude", "", "yes", "Decimal degrees, range -180 to 180"])
    ws_plant.append(["state_or_region", "", "no", "Optional state/region"])
    ws_plant.append(["country", "", "no", "Optional country"])

    df_template = _build_template_dataframe(dc_ac_ratios)
    for row in dataframe_to_rows(df_template, index=False, header=True):
        ws_hourly.append(row)

    ws_hourly.freeze_panes = "A2"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    return output_path


def _read_plant_info(template_path: Path) -> Dict[str, object]:
    try:
        df_plant = pd.read_excel(template_path, sheet_name="plant_info")
    except ValueError:
        return {}

    expected_cols = {"field", "value"}
    if not expected_cols.issubset(df_plant.columns):
        return {}

    meta: Dict[str, object] = {}
    for _, row in df_plant.iterrows():
        field = str(row.get("field", "")).strip()
        if not field:
            continue
        value = row.get("value")
        if pd.isna(value):
            continue
        meta[field] = value

    for numeric_field in ("latitude", "longitude"):
        if numeric_field in meta:
            try:
                meta[numeric_field] = float(meta[numeric_field])
            except (TypeError, ValueError):
                raise ValueError(f"Invalid {numeric_field} in plant_info sheet: {meta[numeric_field]}")

    if "latitude" in meta and not (-90.0 <= float(meta["latitude"]) <= 90.0):
        raise ValueError("Latitude must be in range -90 to 90")
    if "longitude" in meta and not (-180.0 <= float(meta["longitude"]) <= 180.0):
        raise ValueError("Longitude must be in range -180 to 180")

    return meta


def _extract_ratio_columns(columns: Iterable[str]) -> List[Tuple[float, str]]:
    ratio_columns: List[Tuple[float, str]] = []
    for column in columns:
        if not column.startswith(SOLAR_PREFIX):
            continue

        raw_token = column[len(SOLAR_PREFIX) :].strip()
        token = raw_token.replace("_", ".")
        try:
            ratio = float(token)
        except ValueError:
            continue
        ratio_columns.append((ratio, column))

    ratio_columns.sort(key=lambda item: item[0])
    return ratio_columns


def _validate_hourly_dataframe(df_hourly: pd.DataFrame) -> None:
    required = {"hour_index", "hour_of_day", "wind_kwh_per_wtg"}
    missing = sorted(required - set(df_hourly.columns))
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    if len(df_hourly) != HOURS_PER_YEAR:
        raise ValueError(
            f"Expected {HOURS_PER_YEAR} rows in hourly_profiles sheet, found {len(df_hourly)}"
        )


def convert_template_to_profiles(
    template_path: Path,
    output_dir: Path,
    sheet_name: str = "hourly_profiles",
    default_ratio: float | None = None,
) -> List[Path]:
    df_hourly = pd.read_excel(template_path, sheet_name=sheet_name)
    _validate_hourly_dataframe(df_hourly)
    plant_info = _read_plant_info(template_path)

    ratio_columns = _extract_ratio_columns(df_hourly.columns)
    if not ratio_columns:
        raise ValueError(
            "No solar ratio columns found. Add columns like "
            f"{SOLAR_PREFIX}1.4, {SOLAR_PREFIX}1.45"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    by_ratio: Dict[float, Path] = {}

    for ratio, solar_column in ratio_columns:
        out_df = pd.DataFrame(
            {
                "hour_index": df_hourly["hour_index"].astype(int),
                "hour_of_day": df_hourly["hour_of_day"].astype(int),
                "solar_kwh_per_mw": pd.to_numeric(df_hourly[solar_column], errors="coerce").fillna(0.0),
                "dc_ac_ratio": ratio,
                "wind_kwh_per_wtg": pd.to_numeric(df_hourly["wind_kwh_per_wtg"], errors="coerce").fillna(0.0),
            }
        )

        out_file = output_dir / f"generation_profiles_dcac_{_ratio_filename_token(ratio)}.csv"
        out_df.to_csv(out_file, index=False)
        written.append(out_file)
        by_ratio[ratio] = out_file

    if default_ratio is None:
        default_ratio = 1.4

    if default_ratio in by_ratio:
        default_source = by_ratio[default_ratio]
    else:
        default_source = written[0]

    default_target = output_dir / "generation_profiles.csv"
    pd.read_csv(default_source).to_csv(default_target, index=False)
    written.append(default_target)

    if plant_info:
        metadata_target = output_dir / "plant_location.json"
        with open(metadata_target, "w", encoding="utf-8") as fp:
            json.dump(plant_info, fp, indent=2)
        written.append(metadata_target)

    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create or convert multi-dc_ac_ratio generation profile templates"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create-template", help="Create empty Excel template")
    create_parser.add_argument(
        "--output",
        default="standalone_data/generation_profiles_template.xlsx",
        help="Output template workbook path",
    )
    create_parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[1.4, 1.45],
        help="dc_ac_ratio values to include as solar columns",
    )

    convert_parser = subparsers.add_parser("convert", help="Convert template workbook to CSV profiles")
    convert_parser.add_argument("--input", required=True, help="Template workbook path")
    convert_parser.add_argument(
        "--output-dir",
        default="standalone_data",
        help="Directory for generation_profiles*.csv outputs",
    )
    convert_parser.add_argument(
        "--sheet",
        default="hourly_profiles",
        help="Worksheet name containing hourly data",
    )
    convert_parser.add_argument(
        "--default-ratio",
        type=float,
        default=1.4,
        help="Ratio to mirror into generation_profiles.csv when available",
    )

    args = parser.parse_args()

    if args.command == "create-template":
        output = create_template(Path(args.output), args.ratios)
        print(f"Template created: {output}")
        return 0

    if args.command == "convert":
        written_files = convert_template_to_profiles(
            template_path=Path(args.input),
            output_dir=Path(args.output_dir),
            sheet_name=args.sheet,
            default_ratio=args.default_ratio,
        )
        print("Wrote files:")
        for path in written_files:
            print(f"  - {path}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
