#!/usr/bin/env python3
"""Helpers for selecting generation profile files for standalone runs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _read_profile_ratio(profile_path: Path) -> Optional[float]:
    """Return dc_ac_ratio from the profile file if present."""
    try:
        ratio_series = pd.read_csv(profile_path, usecols=["dc_ac_ratio"], nrows=16)["dc_ac_ratio"].dropna()
    except Exception:
        return None

    if ratio_series.empty:
        return None

    try:
        return float(ratio_series.iloc[0])
    except (TypeError, ValueError):
        return None


def select_generation_profile_file(
    data_dir: Path,
    dc_ac_ratio: float,
    explicit_profile: Optional[Path] = None,
    tolerance: float = 1e-9,
) -> Path:
    """
    Select generation profile CSV for a given dc_ac_ratio.

    Rules:
    1. If explicit_profile is provided, use it.
    2. Look for generation_profiles*.csv in data_dir and match dc_ac_ratio column.
    3. Fallback to data_dir/generation_profiles.csv if it exists.
    4. Otherwise fallback to the first generation_profiles*.csv file.
    """
    if explicit_profile is not None:
        return explicit_profile

    candidates = sorted(data_dir.glob("generation_profiles*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No generation profile files found in {data_dir}. Expected generation_profiles*.csv"
        )

    for candidate in candidates:
        candidate_ratio = _read_profile_ratio(candidate)
        if candidate_ratio is not None and abs(candidate_ratio - dc_ac_ratio) <= tolerance:
            return candidate

    default_profile = data_dir / "generation_profiles.csv"
    if default_profile.exists():
        return default_profile

    return candidates[0]
