#!/usr/bin/env python3
"""
Streamlit web UI for Hybrid RE System Simulation & Optimization.
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import tempfile
import io
import time
import shutil
from dataclasses import asdict

from hourly_sim_skeleton import Config, simulate_hourly, assign_tod
from optimizer import (
    OptimizationConfig, OptimizationResult,
    generate_configurations, evaluate_configuration
)
from lcoe_calculator import CostParameters, FinancialParameters
from utils.generation_profiles_template import create_template, convert_template_to_profiles

# Page config
st.set_page_config(
    page_title="Hybrid RE System Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None
if 'optimization_running' not in st.session_state:
    st.session_state.optimization_running = False
if 'cfg' not in st.session_state:
    st.session_state.cfg = None
if 'df_profiles' not in st.session_state:
    st.session_state.df_profiles = None
if 'plant_location' not in st.session_state:
    st.session_state.plant_location = None


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_generation_profile_catalog(data_path: Path):
    catalog_path = data_path / "generation_profile_catalog.json"
    if not catalog_path.exists():
        return []
    try:
        with open(catalog_path) as fp:
            payload = json.load(fp)
        if isinstance(payload, dict) and isinstance(payload.get("locations"), list):
            return payload["locations"]
    except Exception:
        return []
    return []


def _select_profile_from_preset(data_path: Path, preset: dict, dc_ac_ratio: float):
    profiles = preset.get("profiles", {}) if isinstance(preset, dict) else {}
    if not isinstance(profiles, dict) or not profiles:
        return None

    ratio_map = []
    for ratio_key, filename in profiles.items():
        ratio = _safe_float(ratio_key)
        if ratio is None:
            continue
        path = data_path / str(filename)
        if path.exists():
            ratio_map.append((ratio, path))

    if not ratio_map:
        return None

    for ratio, path in ratio_map:
        if abs(ratio - dc_ac_ratio) <= 1e-9:
            return path

    return None


def _select_profile_exact_from_dir(data_path: Path, dc_ac_ratio: float):
    """Return matching generation profile file for exact dc_ac_ratio, else None."""
    for candidate in sorted(data_path.glob("generation_profiles*.csv")):
        try:
            ratio_series = pd.read_csv(candidate, usecols=["dc_ac_ratio"], nrows=16)["dc_ac_ratio"].dropna()
        except Exception:
            continue
        if ratio_series.empty:
            continue
        try:
            ratio = float(ratio_series.iloc[0])
        except (TypeError, ValueError):
            continue
        if abs(ratio - dc_ac_ratio) <= 1e-9:
            return candidate
    return None

# Title and intro
st.title("⚡ Hybrid Renewable Energy System Optimizer")
st.markdown("Simulate and optimize solar/wind/BESS systems for minimum LCOE")

# Sidebar - Data Source Selection
st.sidebar.header("📁 Data Source")

data_source = st.sidebar.radio(
    "Choose data source:",
    ["Use Demo Data", "Upload Excel File", "Use Local Files"],
    help="Demo: Try the tool with sample data\nUpload: Extract data from your Excel\nLocal: Use pre-extracted standalone_data/"
)

files_exist = False
default_config = None

# Handle different data sources
if data_source == "Use Demo Data":
    st.sidebar.info("🎯 Using demo configuration (830 MW solar, 50 MW BESS)")
    
    # Check if demo data exists locally
    demo_path = Path("standalone_data")
    if demo_path.exists():
        data_path = demo_path
        config_file = demo_path / "system_config.json"
        load_file = demo_path / "load_profile.csv"
        financial_file = demo_path / "financial_params.json"
        generation_profile_files = list(data_path.glob("generation_profiles*.csv"))
        files_exist = all([f.exists() for f in [config_file, load_file, financial_file]]) and bool(generation_profile_files)
    
    if not files_exist:
        st.warning("⚠️ Demo data not available. Please upload an Excel file or use local files.")
        st.sidebar.markdown("**Deployment Note**: Include `standalone_data/` in your repository for demo mode.")
        st.stop()
    
    loc_file = data_path / "plant_location.json"
    if loc_file.exists():
        with open(loc_file) as _lf:
            st.session_state.plant_location = json.load(_lf)
    else:
        st.session_state.plant_location = None
    st.session_state.data_loaded = True

elif data_source == "Upload Excel File":
    st.sidebar.markdown("Upload your Excel file to extract data")

    template_path = Path("standalone_data") / "generation_profiles_template.xlsx"
    if not template_path.exists():
        try:
            create_template(template_path, [1.4, 1.45])
        except Exception:
            template_path = None

    if template_path and template_path.exists():
        with open(template_path, "rb") as template_file:
            st.sidebar.download_button(
                "Download Generation Template",
                data=template_file.read(),
                file_name="generation_profiles_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download template, fill hourly_profiles data, then upload below."
            )
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose Excel file", 
        type=['xlsx', 'xls'],
        key="full_model_upload",
        help="Upload your hybrid RE system Excel model"
    )

    uploaded_template_file = st.sidebar.file_uploader(
        "Upload Filled Generation Template",
        type=['xlsx'],
        key="generation_template_upload",
        help="Upload the downloaded generation template after filling hourly profiles."
    )
    
    if uploaded_file is not None:
        with st.spinner("Extracting data from Excel..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                # Extract data
                from utils.extract_standalone_data import extract_standalone_data
                extract_dir = tempfile.mkdtemp()
                extract_standalone_data(tmp_path, extract_dir)
                
                # Update paths
                data_path = Path(extract_dir)
                config_file = data_path / "system_config.json"
                load_file = data_path / "load_profile.csv"
                financial_file = data_path / "financial_params.json"
                
                files_exist = True
                st.sidebar.success("✅ Data extracted successfully!")
                st.session_state.data_loaded = True
                
            except Exception as e:
                st.sidebar.error(f"Error extracting data: {e}")
                st.stop()
    elif uploaded_template_file is not None:
        with st.spinner("Converting generation template..."):
            try:
                base_data_path = Path("standalone_data")
                base_config_file = base_data_path / "system_config.json"
                base_load_file = base_data_path / "load_profile.csv"
                base_financial_file = base_data_path / "financial_params.json"

                if not all([base_config_file.exists(), base_load_file.exists(), base_financial_file.exists()]):
                    st.sidebar.error(
                        "Missing baseline files in standalone_data. Expected system_config.json, "
                        "load_profile.csv, and financial_params.json."
                    )
                    st.stop()

                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_template_file:
                    tmp_template_file.write(uploaded_template_file.read())
                    tmp_template_path = Path(tmp_template_file.name)

                extract_dir = Path(tempfile.mkdtemp())
                convert_template_to_profiles(
                    template_path=tmp_template_path,
                    output_dir=extract_dir,
                    default_ratio=1.4,
                )

                shutil.copy2(base_config_file, extract_dir / "system_config.json")
                shutil.copy2(base_load_file, extract_dir / "load_profile.csv")
                shutil.copy2(base_financial_file, extract_dir / "financial_params.json")

                data_path = extract_dir
                config_file = data_path / "system_config.json"
                load_file = data_path / "load_profile.csv"
                financial_file = data_path / "financial_params.json"

                loc_file = data_path / "plant_location.json"
                if loc_file.exists():
                    with open(loc_file) as _lf:
                        st.session_state.plant_location = json.load(_lf)
                else:
                    st.session_state.plant_location = None

                files_exist = True
                st.sidebar.success("✅ Template converted successfully!")
                st.session_state.data_loaded = True

            except Exception as e:
                st.sidebar.error(f"Error converting template: {e}")
                st.stop()
    else:
        st.info("👈 Upload an Excel file to get started")
        st.markdown("""
        ### How to use:
        1. Upload your hybrid RE Excel model
        2. Or download generation template, fill it, and upload in the same panel
        2. Configure system parameters (solar, wind, BESS)
        3. Click "Run Simulation" to see results
        
        **Don't have an Excel file?** Switch to "Use Demo Data" to try it out!
        """)
        st.stop()

else:  # Use Local Files
    data_dir = st.sidebar.text_input("Data Directory", "standalone_data")
    data_path = Path(data_dir)
    config_file = data_path / "system_config.json"
    load_file = data_path / "load_profile.csv"
    financial_file = data_path / "financial_params.json"
    generation_profile_files = list(data_path.glob("generation_profiles*.csv"))
    
    files_exist = all([f.exists() for f in [config_file, load_file, financial_file]]) and bool(generation_profile_files)
    
    if not files_exist:
        st.sidebar.error("⚠️ Data files not found!")
        st.sidebar.markdown(f"""
        Required files in `{data_dir}`:
        - system_config.json
        - generation_profiles*.csv
        - load_profile.csv
        - financial_params.json
        
        Run: `python utils/extract_standalone_data.py your_file.xlsx`
        """)
        st.stop()
    else:
        st.sidebar.success("✓ Data files loaded")
        loc_file = data_path / "plant_location.json"
        if loc_file.exists():
            with open(loc_file) as _lf:
                st.session_state.plant_location = json.load(_lf)
        else:
            st.session_state.plant_location = None
        st.session_state.data_loaded = True

# Load default configuration
with open(config_file) as f:
    default_config = json.load(f)

# Generation profile selection controls
uploaded_generation_profile = None
uploaded_generation_profile_main = None
selected_preset = None
catalog_locations = _load_generation_profile_catalog(data_path)

# --- Location selector on main screen ---
preset_names = [str(item.get("name", "")).strip() for item in catalog_locations if str(item.get("name", "")).strip()]
location_options = preset_names + ["Location not available"]

st.subheader("📍 Select Location")
selected_location = st.selectbox(
    "Location",
    location_options,
    label_visibility="collapsed",
    help="Choose a preset location or select 'Location not available' to upload your own generation profile."
)

if selected_location == "Location not available":
    profile_selection_mode = "Upload New Location CSV"
    st.info("No preset profile for your location. Upload a generation profile CSV below.")
    uploaded_generation_profile = st.file_uploader(
        "Upload generation profile CSV",
        type=["csv"],
        key="uploaded_generation_profile_csv",
        help="Upload a generation profile CSV for your location."
    )
else:
    profile_selection_mode = "Preset Location"
    selected_preset = next((item for item in catalog_locations if item.get("name") == selected_location), None)
    if selected_preset:
        _plat = selected_preset.get("latitude")
        _plon = selected_preset.get("longitude")
        st.caption(f"Coordinates: {_plat}° N, {_plon}° E")

st.divider()

# Main tabs
tab1, tab2, tab3 = st.tabs(["🔧 Configuration", "📊 Simulation Results", "🎯 Optimization"])

with tab1:
    st.header("System Configuration")

    _loc = st.session_state.get("plant_location")
    if selected_preset:
        _loc = {
            "plant_name": selected_preset.get("name", ""),
            "place_name": selected_preset.get("name", ""),
            "latitude": selected_preset.get("latitude", ""),
            "longitude": selected_preset.get("longitude", ""),
        }
    if _loc:
        _name = _loc.get("plant_name") or _loc.get("place_name", "")
        _place = _loc.get("place_name", "")
        _lat = _loc.get("latitude", "")
        _lon = _loc.get("longitude", "")
        _region = _loc.get("state_or_region", "")
        _country = _loc.get("country", "")
        _location_parts = [p for p in [_place, _region, _country] if p]
        _location_str = ", ".join(str(p) for p in _location_parts) if _location_parts else "—"
        _coords_str = f"{_lat}° N, {_lon}° E" if _lat and _lon else "—"
        with st.container(border=True):
            st.markdown(f"**📍 Plant: {_name}**")
            loc_c1, loc_c2 = st.columns(2)
            loc_c1.markdown(f"**Location:** {_location_str}")
            loc_c2.markdown(f"**Coordinates:** {_coords_str}")
        st.divider()

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🌞 Solar")
        solar_ac_mw = st.number_input(
            "Solar Capacity (MW AC)", 
            min_value=0.0, 
            max_value=5000.0, 
            value=float(default_config.get('solar_ac_mw', 830.0)),
            step=10.0
        )
        dc_ac_ratio = st.number_input(
            "DC/AC Ratio",
            min_value=1.0,
            max_value=2.0,
            value=float(default_config.get('dc_ac_ratio', 1.45)),
            step=0.05
        )
        solar_degrad = st.number_input(
            "Degradation (%/year)",
            min_value=0.0,
            max_value=2.0,
            value=float(default_config.get('solar_degrad', 0.005)) * 100,
            step=0.1,
            key="solar_degrad"
        ) / 100
    
    with col2:
        st.subheader("💨 Wind")
        wind_wtg_count = st.number_input(
            "Number of WTGs",
            min_value=0,
            max_value=200,
            value=int(default_config.get('wind_wtg_count', 0)),
            step=1
        )
        wind_degrad = st.number_input(
            "Degradation (%/year)",
            min_value=0.0,
            max_value=2.0,
            value=float(default_config.get('wind_degrad', 0.005)) * 100,
            step=0.1,
            key="wind_degrad"
        ) / 100
    
    with col3:
        st.subheader("🔋 BESS")
        bess_power_mw = st.number_input(
            "Power (MW)",
            min_value=0.0,
            max_value=500.0,
            value=float(default_config.get('bess_power_mw', 50.0)),
            step=5.0
        )
        bess_energy_mwh = st.number_input(
            "Energy (MWh)",
            min_value=0.0,
            max_value=2000.0,
            value=float(default_config.get('bess_energy_mwh', 200.0)),
            step=10.0
        )
        bess_eff = st.number_input(
            "One-way Efficiency",
            min_value=0.5,
            max_value=1.0,
            value=float(default_config.get('one_way_eff', 0.9487)),
            step=0.01,
            format="%.4f"
        )
    
    st.subheader("⚙️ System Parameters")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        total_load_mw = st.number_input(
            "Load (MW)",
            min_value=0.0,
            max_value=5000.0,
            value=float(default_config.get('total_load_mw', 800.0)),
            step=10.0
        )
    
    with col5:
        contracted_demand_mw = st.number_input(
            "Contracted Demand (MW)",
            min_value=0.0,
            max_value=5000.0,
            value=float(default_config.get('contracted_demand_mw', 800.0)),
            step=10.0
        )
    
    with col6:
        evac_limit_mw = st.number_input(
            "Evacuation Limit (MW)",
            min_value=0.0,
            max_value=5000.0,
            value=float(default_config.get('evac_limit_mw', 800.0)),
            step=10.0
        )

    st.subheader("📍 Location")
    col7, col8, col9 = st.columns(3)

    location_default_name = str(default_config.get('location_name', ''))
    location_default_lat = float(default_config.get('latitude', 0.0) or 0.0)
    location_default_lon = float(default_config.get('longitude', 0.0) or 0.0)
    if _loc:
        location_default_name = str(_loc.get('plant_name') or _loc.get('place_name') or location_default_name)
        loc_lat = _safe_float(_loc.get('latitude'))
        loc_lon = _safe_float(_loc.get('longitude'))
        location_default_lat = loc_lat if loc_lat is not None else location_default_lat
        location_default_lon = loc_lon if loc_lon is not None else location_default_lon

    with col7:
        location_name = st.text_input(
            "Location Name",
            value=location_default_name,
            help="Plant/site name"
        )

    with col8:
        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=location_default_lat,
            step=0.0001,
            format="%.6f"
        )

    with col9:
        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=location_default_lon,
            step=0.0001,
            format="%.6f"
        )

    st.subheader("📥 New Location Profile")
    st.caption("If your location is not in the preset dropdown, upload a generation profile CSV here.")
    uploaded_generation_profile_main = st.file_uploader(
        "Upload generation profile CSV (main page)",
        type=["csv"],
        key="uploaded_generation_profile_main_csv",
        help="This uploaded file takes priority over preset selection for this run."
    )
    
    st.subheader("💰 Financial Parameters")
    project_lifetime_years = st.slider(
        "Project Lifetime (Years)",
        min_value=10,
        max_value=30,
        value=25,
        step=1,
        help="Number of years for LCOE calculation"
    )
    
    # Run simulation button
    if st.button("▶️ Run Simulation", type="primary", width='stretch'):
        with st.spinner("Running simulation..."):
            # Create config
            from dataclasses import fields
            valid_fields = {f.name for f in fields(Config)}
            filtered_default = {k: v for k, v in default_config.items() if k in valid_fields}
            
            # Override with user inputs
            user_inputs = {
                'solar_ac_mw': float(solar_ac_mw),
                'dc_ac_ratio': float(dc_ac_ratio),
                'solar_degrad': float(solar_degrad),
                'wind_wtg_count': float(wind_wtg_count),
                'wind_degrad': float(wind_degrad),
                'bess_power_mw': float(bess_power_mw),
                'bess_energy_mwh': float(bess_energy_mwh),
                'one_way_eff': float(bess_eff),
                'total_load_mw': float(total_load_mw),
                'contracted_demand_mw': float(contracted_demand_mw),
                'evac_limit_mw': float(evac_limit_mw),
                'location_name': str(location_name),
                'latitude': float(latitude),
                'longitude': float(longitude),
            }
            filtered_default.update(user_inputs)
            
            cfg = Config(**filtered_default)
            
            # Load profiles
            if uploaded_generation_profile_main is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_profile_file:
                    tmp_profile_file.write(uploaded_generation_profile_main.read())
                    profiles_file = Path(tmp_profile_file.name)
            elif profile_selection_mode == "Upload New Location CSV":
                if uploaded_generation_profile is None:
                    st.error("Please upload a generation profile CSV or switch to preset location.")
                    st.stop()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_profile_file:
                    tmp_profile_file.write(uploaded_generation_profile.read())
                    profiles_file = Path(tmp_profile_file.name)
            elif selected_preset:
                preset_profile = _select_profile_from_preset(data_path, selected_preset, cfg.dc_ac_ratio)
                if preset_profile is None:
                    st.error(
                        f"No generation profile data available for location '{selected_preset.get('name')}' "
                        f"at dc_ac_ratio={cfg.dc_ac_ratio}. Please choose another ratio or upload a custom profile."
                    )
                    st.stop()
                profiles_file = preset_profile
            else:
                exact_profile = _select_profile_exact_from_dir(data_path, cfg.dc_ac_ratio)
                if exact_profile is None:
                    st.error(
                        f"No generation profile data available for dc_ac_ratio={cfg.dc_ac_ratio}. "
                        "Please choose another ratio or upload a custom profile."
                    )
                    st.stop()
                profiles_file = exact_profile

            df_gen = pd.read_csv(profiles_file)
            df_load = pd.read_csv(load_file)

            if "dc_ac_ratio" in df_gen.columns:
                ratio_vals = pd.to_numeric(df_gen["dc_ac_ratio"], errors="coerce").dropna()
                if not ratio_vals.empty:
                    file_ratio = float(ratio_vals.iloc[0])
                    if abs(file_ratio - cfg.dc_ac_ratio) > 1e-9:
                        st.error(
                            f"Uploaded/selected profile has dc_ac_ratio={file_ratio}, "
                            f"but configuration is dc_ac_ratio={cfg.dc_ac_ratio}."
                        )
                        st.stop()

            if "location_name" in df_gen.columns:
                loc_name = df_gen["location_name"].dropna().astype(str)
                if not loc_name.empty and not location_name:
                    location_name = loc_name.iloc[0]
            if "latitude" in df_gen.columns:
                lat_vals = pd.to_numeric(df_gen["latitude"], errors="coerce").dropna()
                if not lat_vals.empty and latitude == 0.0:
                    latitude = float(lat_vals.iloc[0])
            if "longitude" in df_gen.columns:
                lon_vals = pd.to_numeric(df_gen["longitude"], errors="coerce").dropna()
                if not lon_vals.empty and longitude == 0.0:
                    longitude = float(lon_vals.iloc[0])

            cfg.location_name = str(location_name)
            cfg.latitude = float(latitude)
            cfg.longitude = float(longitude)
            
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
            
            # Store profiles in session state for optimization
            st.session_state.df_profiles = df_profiles
            
            # Run simulation for Year 1
            df_sim = simulate_hourly(cfg, df_profiles)
            
            # Store Year 1 results for visualization
            st.session_state.df_sim = df_sim
            st.session_state.cfg = cfg
            
            # Calculate LCOE
            try:
                from lcoe_calculator import (
                    CostParameters, FinancialParameters,
                    calculate_lcoe as calc_lcoe_func
                )
                
                # Load financial parameters from JSON
                with open(financial_file) as f:
                    fin_data = json.load(f)
                
                # Use actual costs from financial params (extracted from Excel)
                project_cost = fin_data.get('project_cost_crore', 3949.56)
                annual_opex = fin_data.get('annual_opex_crore', 46.48)
                
                # Cost parameters (only needed for residual value setting)
                cost_params = CostParameters(
                    residual_value_fraction=0.0  # 0% residual value to match Excel
                )
                
                # Financial parameters
                fin_params = FinancialParameters(
                    tax_rate=fin_data.get('tax_rate', 0.2782),
                    discount_rate=fin_data.get('discount_rate', 0.0749997),
                    interest_rate=fin_data.get('loan_interest_rate', 0.095),
                    system_degradation=cfg.solar_degrad,
                    opex_escalation=0.05,  # 5% annual OPEX escalation
                    equity_fraction=fin_data.get('equity_percent', 0.3),
                    loan_fraction=fin_data.get('loan_percent', 0.7),
                    loan_term_years=int(fin_data.get('loan_term_years', 10)),
                    depreciation_rate_early=0.04666666,  # 4.67% for years 1-15
                    depreciation_rate_late=0.03,          # 3% for years 16+
                    depreciation_switchover_year=15,
                    project_lifetime_years=project_lifetime_years  # Use slider value
                )
                
                # Calculate multi-year energy delivery with degradation
                # Year 1 energy (MWh)
                year1_energy_mwh = df_sim['delivered_total'].sum()
                year1_energy_kwh = year1_energy_mwh * 1000  # Convert to kWh
                
                # Pass constant Year 1 energy - calculate_lcoe will apply degradation internally
                annual_energy_kwh = np.full(project_lifetime_years, year1_energy_kwh)
                
                # Also pre-calculate degraded energy for display purposes
                annual_energy_degraded_kwh = np.zeros(project_lifetime_years)
                for year_idx in range(project_lifetime_years):
                    # Note: calculate_lcoe uses (1 - system_degradation) ** (n - 1) where n = year + 1
                    # So year 0 (n=1) has factor (1-deg)^0 = 1, year 1 (n=2) has factor (1-deg)^1, etc.
                    degradation_factor = (1 - cfg.solar_degrad) ** year_idx
                    annual_energy_degraded_kwh[year_idx] = year1_energy_kwh * degradation_factor
                
                # Calculate LCOE
                lcoe_results = calc_lcoe_func(
                    project_cost, 
                    annual_opex, 
                    annual_energy_kwh, 
                    fin_params, 
                    cost_params
                )
                lcoe = lcoe_results['lcoe_inr_per_kwh']
                
                # Store additional multi-year metrics (use degraded energy for display)
                st.session_state.annual_energy_kwh = annual_energy_degraded_kwh
                st.session_state.total_lifetime_energy_gwh = annual_energy_degraded_kwh.sum() / 1e9
                st.session_state.project_lifetime_years = project_lifetime_years
                
            except Exception as e:
                lcoe = None
                st.warning(f"Could not calculate LCOE: {e}")
            
            # Store parameters needed for optimization
            st.session_state.project_lifetime_years_slider = project_lifetime_years
            st.session_state.profiles_file = profiles_file
            st.session_state.load_file = load_file
            st.session_state.financial_file = financial_file
            
            # Store final results in session
            st.session_state.lcoe = lcoe
            st.session_state.simulation_run = True
            
        st.success("✅ Simulation complete! Click on the **📊 Simulation Results** tab above to see full details.")
        
        # Show quick preview
        st.subheader("Quick Results Preview (Year 1)")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Solar Generation", f"{df_sim['solar_gen'].sum():,.0f} MWh")
        with col2:
            st.metric("Wind Generation", f"{df_sim['wind_gen'].sum():,.0f} MWh")
        with col3:
            st.metric("BESS Discharge", f"{df_sim['bess_discharge_inj'].sum():,.0f} MWh")
        with col4:
            st.metric("Grid Import", f"{df_sim['from_grid'].sum():,.0f} MWh")
        with col5:
            if lcoe:
                st.metric("LCOE", f"₹{lcoe:.4f}/kWh", help=f"Based on {project_lifetime_years}-year projection")
            else:
                st.metric("LCOE", "N/A")

with tab2:
    if not st.session_state.simulation_run:
        st.info("👈 Configure system and run simulation to see results")
    else:
        df_sim = st.session_state.df_sim
        cfg = st.session_state.cfg
        lcoe = st.session_state.lcoe
        project_years = st.session_state.get('project_lifetime_years', 25)
        total_lifetime_gwh = st.session_state.get('total_lifetime_energy_gwh', 0)
        
        st.header("Simulation Results")
        
        # Project Summary with LCOE
        col_summary1, col_summary2, col_summary3 = st.columns(3)
        
        with col_summary1:
            st.metric("📅 Project Lifetime", f"{project_years} years", help="Total project duration")
        with col_summary2:
            st.metric("⚡ Lifetime Energy", f"{total_lifetime_gwh:,.1f} GWh", help="Total energy delivered over project lifetime")
        with col_summary3:
            if lcoe:
                st.metric("💰 LCOE", f"₹{lcoe:.4f}/kWh", help=f"Levelized Cost of Energy over {project_years} years")
            else:
                st.metric("💰 LCOE", "N/A", help="Could not calculate LCOE")
        
        st.divider()
        
        # Key metrics (Year 1)
        st.markdown("### Year 1 Results")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Solar Generation", f"{df_sim['solar_gen'].sum():,.0f} MWh")
        with col2:
            st.metric("Wind Generation", f"{df_sim['wind_gen'].sum():,.0f} MWh")
        with col3:
            st.metric("BESS Discharge", f"{df_sim['bess_discharge_inj'].sum():,.0f} MWh")
        with col4:
            st.metric("Grid Import", f"{df_sim['from_grid'].sum():,.0f} MWh")
        with col5:
            st.metric("Total Delivered", f"{df_sim['delivered_total'].sum():,.0f} MWh")
        
        # Detailed metrics table
        st.subheader("📋 Detailed Metrics")
        metrics_data = {
            "Metric": [
                "Total Generation",
                "To Consumption",
                "Curtailment",
                "BESS Charge",
                "BESS Discharge",
                "Direct Delivered",
                "From BESS",
                "From Grid",
                "Total Delivered"
            ],
            "Value (MWh)": [
                df_sim['gen_total'].sum(),
                df_sim['to_consumption_inj'].sum(),
                df_sim['curtailed'].sum(),
                df_sim['bess_charge_inj'].sum(),
                df_sim['bess_discharge_inj'].sum(),
                df_sim['direct_delivered'].sum(),
                df_sim['from_bess_delivered'].sum(),
                df_sim['from_grid'].sum(),
                df_sim['delivered_total'].sum(),
            ]
        }
        st.dataframe(pd.DataFrame(metrics_data), width='stretch', hide_index=True)
        
        # Multi-Year Analysis
        st.subheader(f"📊 Multi-Year Analysis ({project_years} Years)")
        
        # Calculate metrics for all years
        if 'annual_energy_kwh' in st.session_state:
            # Year 1 base metrics (MWh)
            year1_solar = df_sim['solar_gen'].sum()
            year1_wind = df_sim['wind_gen'].sum()
            year1_bess_discharge = df_sim['bess_discharge_inj'].sum()
            year1_grid = df_sim['from_grid'].sum()
            year1_delivered = df_sim['delivered_total'].sum()
            year1_curtailed = df_sim['curtailed'].sum()
            
            # Build multi-year dataframe
            years_list = []
            for year_num in range(1, project_years + 1):
                deg_factor = (1 - cfg.solar_degrad) ** (year_num - 1)
                years_list.append({
                    'Year': year_num,
                    'Solar (GWh)': year1_solar * deg_factor / 1000,
                    'Wind (GWh)': year1_wind * deg_factor / 1000,
                    'BESS Discharge (GWh)': year1_bess_discharge * deg_factor / 1000,
                    'Grid Import (GWh)': year1_grid * deg_factor / 1000,
                    'Delivered (GWh)': year1_delivered * deg_factor / 1000,
                    'Curtailed (GWh)': year1_curtailed * deg_factor / 1000,
                    'Degradation Factor': f"{deg_factor:.4f}"
                })
            
            df_multiyear = pd.DataFrame(years_list)
            
            # Show table with option to expand
            with st.expander(f"📋 Annual Summary Table (All {project_years} Years)", expanded=False):
                st.dataframe(df_multiyear, width='stretch', hide_index=True, height=400)
                
                # Download button
                csv = df_multiyear.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"multiyear_results_{project_years}years.csv",
                    mime="text/csv"
                )
            
            # Financial Summary
            if lcoe:
                with st.expander("💰 Financial Summary & LCOE Breakdown", expanded=False):
                    st.markdown(f"""
                    ### LCOE: ₹{lcoe:.4f} per kWh
                    
                    **Key Financial Parameters:**
                    - **Project Cost**: ₹3,949.56 Crore
                    - **Annual OPEX**: ₹46.48 Crore (Year 1, escalates at 5%/year)
                    - **Project Lifetime**: {project_years} years
                    - **Discount Rate**: 7.50%
                    - **Tax Rate**: 27.82%
                    - **System Degradation**: {cfg.solar_degrad*100:.2f}% per year
                    - **OPEX Escalation**: 5.0% per year
                    
                    **Energy Delivery:**
                    - **Year 1 Energy**: {year1_delivered:,.0f} MWh ({year1_delivered/1000:.2f} GWh)
                    - **Lifetime Energy**: {total_lifetime_gwh:,.1f} GWh
                    - **Average Annual Energy**: {total_lifetime_gwh/project_years:.2f} GWh
                    
                    ---
                    
                    **💡 How LCOE Changes with Project Lifetime:**
                    
                    Longer project lifetimes generally result in **lower LCOE** because:
                    - Capital cost is spread over more energy delivered
                    - However, degradation reduces later years' generation
                    - OPEX escalation increases costs in later years
                    - Discounting reduces value of distant cash flows
                    
                    *Try adjusting the "Project Lifetime" slider in the Configuration tab to see the impact!*
                    
                    *LCOE represents the total lifecycle cost divided by total energy delivered, considering time value of money.*
                    """)
                    
                    # Show how LCOE varies with project lifetime
                    st.markdown("#### 📊 LCOE Sensitivity to Project Lifetime")
                    
                    # Calculate LCOE for different project lifetimes
                    from lcoe_calculator import (
                        CostParameters, FinancialParameters,
                        calculate_lcoe as calc_lcoe_func
                    )
                    
                    test_years = [10, 15, 20, 25, 30]
                    lcoe_values = []
                    
                    for test_year in test_years:
                        test_fin_params = FinancialParameters(
                            tax_rate=0.2782,
                            discount_rate=0.0749997,
                            interest_rate=0.095,
                            system_degradation=cfg.solar_degrad,
                            opex_escalation=0.05,
                            equity_fraction=0.3,
                            loan_fraction=0.7,
                            loan_term_years=10,
                            project_lifetime_years=test_year
                        )
                        
                        test_annual_energy = np.full(test_year, year1_delivered * 1000)
                        
                        test_results = calc_lcoe_func(
                            3949.56,  # project cost
                            46.48,    # annual opex
                            test_annual_energy,
                            test_fin_params,
                            CostParameters(residual_value_fraction=0.0)
                        )
                        lcoe_values.append(test_results['lcoe_inr_per_kwh'])
                    
                    fig_lcoe_sensitivity = go.Figure()
                    fig_lcoe_sensitivity.add_trace(go.Scatter(
                        x=test_years,
                        y=lcoe_values,
                        mode='lines+markers',
                        name='LCOE',
                        line=dict(color='#FF6B6B', width=3),
                        marker=dict(size=10)
                    ))
                    
                    # Highlight current selection
                    current_lcoe_idx = test_years.index(project_years) if project_years in test_years else None
                    if current_lcoe_idx is not None:
                        fig_lcoe_sensitivity.add_trace(go.Scatter(
                            x=[test_years[current_lcoe_idx]],
                            y=[lcoe_values[current_lcoe_idx]],
                            mode='markers',
                            name='Current Selection',
                            marker=dict(size=15, color='#4ECDC4', symbol='star')
                        ))
                    
                    fig_lcoe_sensitivity.update_layout(
                        title="How LCOE Changes with Project Lifetime",
                        xaxis_title="Project Lifetime (Years)",
                        yaxis_title="LCOE (₹/kWh)",
                        hovermode='x unified',
                        height=350
                    )
                    
                    st.plotly_chart(fig_lcoe_sensitivity, width='stretch')
                    
                    st.caption("💡 **Key Insight**: Longer project lifetimes result in lower LCOE because the upfront capital cost is spread over more energy generation, even accounting for degradation.")
        
        # Charts
        st.subheader("📈 Visualizations")
        
        # LCOE Sensitivity Analysis
        if lcoe and 'annual_energy_kwh' in st.session_state:
            st.markdown("#### LCOE Sensitivity to Project Lifetime")
            
            # Calculate LCOE for different project lifetimes
            from lcoe_calculator import (
                CostParameters, FinancialParameters,
                calculate_lcoe as calc_lcoe_func
            )
            
            test_years = list(range(10, 31))
            lcoe_values = []
            
            # Get financial params
            with open(financial_file) as f:
                fin_data = json.load(f)
            
            project_cost = fin_data.get('project_cost_crore', 3949.56)
            annual_opex = fin_data.get('annual_opex_crore', 46.48)
            cost_params = CostParameters(residual_value_fraction=0.0)
            
            for test_year in test_years:
                fin_params = FinancialParameters(
                    tax_rate=fin_data.get('tax_rate', 0.2782),
                    discount_rate=fin_data.get('discount_rate', 0.0749997),
                    interest_rate=fin_data.get('loan_interest_rate', 0.095),
                    system_degradation=cfg.solar_degrad,
                    opex_escalation=0.05,
                    equity_fraction=fin_data.get('equity_percent', 0.3),
                    loan_fraction=fin_data.get('loan_percent', 0.7),
                    loan_term_years=int(fin_data.get('loan_term_years', 10)),
                    depreciation_rate_early=0.04666666,
                    depreciation_rate_late=0.03,
                    depreciation_switchover_year=15,
                    project_lifetime_years=test_year
                )
                
                year1_energy_kwh = df_sim['delivered_total'].sum() * 1000
                annual_energy_kwh = np.full(test_year, year1_energy_kwh)
                
                lcoe_results = calc_lcoe_func(project_cost, annual_opex, annual_energy_kwh, fin_params, cost_params)
                lcoe_values.append(lcoe_results['lcoe_inr_per_kwh'])
            
            # Plot LCOE sensitivity
            fig_lcoe_sens = go.Figure()
            fig_lcoe_sens.add_trace(go.Scatter(
                x=test_years,
                y=lcoe_values,
                mode='lines+markers',
                name='LCOE',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=6)
            ))
            
            # Highlight current project lifetime
            current_idx = test_years.index(project_years) if project_years in test_years else None
            if current_idx is not None:
                fig_lcoe_sens.add_trace(go.Scatter(
                    x=[project_years],
                    y=[lcoe],
                    mode='markers',
                    name='Current Selection',
                    marker=dict(size=15, color='red', symbol='star')
                ))
            
            fig_lcoe_sens.update_layout(
                title="LCOE vs Project Lifetime",
                xaxis_title="Project Lifetime (Years)",
                yaxis_title="LCOE (₹/kWh)",
                hovermode='x unified',
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig_lcoe_sens, width='stretch')
            
            st.caption(f"💡 Insight: Changing from 10 to 30 years changes LCOE from ₹{lcoe_values[0]:.4f} to ₹{lcoe_values[-1]:.4f} per kWh ({((lcoe_values[-1]/lcoe_values[0])-1)*100:+.1f}%)")
        
        
        # Multi-year trends
        if 'annual_energy_kwh' in st.session_state:
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                # Annual Energy with Degradation
                annual_energy_kwh = st.session_state.annual_energy_kwh
                years = np.arange(1, len(annual_energy_kwh) + 1)
                
                fig_degradation = go.Figure()
                fig_degradation.add_trace(go.Scatter(
                    x=years, 
                    y=annual_energy_kwh / 1e9,  # Convert to GWh
                    name='Annual Energy',
                    fill='tozeroy',
                    line=dict(color='#2E86AB', width=2)
                ))
                fig_degradation.update_layout(
                    title=f"Energy Delivery ({cfg.solar_degrad*100:.2f}% degradation)",
                    xaxis_title="Year",
                    yaxis_title="Energy (GWh)",
                    hovermode='x unified',
                    height=350
                )
                st.plotly_chart(fig_degradation, width='stretch')
            
            with col_viz2:
                # Generation Mix Over Time
                fig_mix = go.Figure()
                years = np.arange(1, project_years + 1)
                
                for year_num in years:
                    deg_factor = (1 - cfg.solar_degrad) ** (year_num - 1)
                
                fig_mix.add_trace(go.Bar(
                    x=years[::max(1, project_years//10)],  # Sample every N years for clarity
                    y=[year1_solar * (1 - cfg.solar_degrad) ** (y - 1) / 1000 for y in years[::max(1, project_years//10)]],
                    name='Solar',
                    marker_color='#FFA500'
                ))
                fig_mix.add_trace(go.Bar(
                    x=years[::max(1, project_years//10)],
                    y=[year1_bess_discharge * (1 - cfg.solar_degrad) ** (y - 1) / 1000 for y in years[::max(1, project_years//10)]],
                    name='BESS',
                    marker_color='#00AA00'
                ))
                fig_mix.add_trace(go.Bar(
                    x=years[::max(1, project_years//10)],
                    y=[year1_grid * (1 - cfg.solar_degrad) ** (y - 1) / 1000 for y in years[::max(1, project_years//10)]],
                    name='Grid',
                    marker_color='#AA0000'
                ))
                fig_mix.update_layout(
                    title="Generation Mix Over Time",
                    xaxis_title="Year",
                    yaxis_title="Energy (GWh)",
                    barmode='stack',
                    height=350,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_mix, width='stretch')
        
        # Detailed Hourly Analysis
        st.subheader("🔍 Detailed Hourly Analysis")
        
        # Year selector
        selected_year = st.selectbox(
            "Select Year to View",
            options=list(range(1, project_years + 1)),
            index=0,
            help="View hourly simulation data for a specific year (degradation applied)"
        )
        
        # Apply degradation to Year 1 data for selected year
        if selected_year > 1:
            deg_factor = (1 - cfg.solar_degrad) ** (selected_year - 1)
            df_year = df_sim.copy()
            # Apply degradation to generation columns
            for col in ['solar_gen', 'wind_gen', 'gen_total', 'to_consumption_inj', 
                       'bess_charge_inj', 'bess_discharge_inj', 'direct_delivered',
                       'from_bess_delivered', 'delivered_total', 'curtailed']:
                if col in df_year.columns:
                    df_year[col] = df_year[col] * deg_factor
            st.info(f"📉 Showing Year {selected_year} data (degradation factor: {deg_factor:.4f})")
        else:
            df_year = df_sim
            st.info(f"📊 Showing Year 1 data (no degradation)")
        
        # Sample week
        week_start = 24 * 7 * 20  # Week 20
        week_end = week_start + 24 * 7
        df_week = df_year.iloc[week_start:week_end]
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df_week.index, y=df_week['solar_gen'], name='Solar', fill='tozeroy'))
        fig1.add_trace(go.Scatter(x=df_week.index, y=df_week['wind_gen'], name='Wind', fill='tozeroy'))
        fig1.add_trace(go.Scatter(x=df_week.index, y=df_week['load_total'], name='Load', line=dict(color='red', dash='dash')))
        fig1.update_layout(title=f"Generation vs Load - Week 20, Year {selected_year}", xaxis_title="Hour", yaxis_title="MW", hovermode='x unified')
        st.plotly_chart(fig1, width='stretch')
        
        # BESS SOC
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_week.index, y=df_week['soc_mwh'], name='SOC', fill='tozeroy'))
        fig2.update_layout(title=f"BESS State of Charge - Week 20, Year {selected_year}", xaxis_title="Hour", yaxis_title="MWh", hovermode='x')
        st.plotly_chart(fig2, width='stretch')

with tab3:
    st.header("🎯 Optimization")
    st.markdown("""
    Find the best solar/wind/BESS configuration to **minimize LCOE** while meeting your load demand.
    
    **Optimization Logic:**
    - **Fixed:** Load demand, evacuation limits, degradation rates (from simulation)
    - **Variable:** Solar capacity, wind capacity, BESS sizing
    - **Objective:** Minimize LCOE
    - **Method:** Grid search across all combinations
    """)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please configure and run a simulation first to enable optimization")
    else:
        # Show fixed parameters from base simulation at the top
        if st.session_state.cfg is not None:
            with st.expander("🔒 **Fixed Parameters** (from your simulation)", expanded=True):
                base_cfg = st.session_state.cfg
                
                col_fix1, col_fix2, col_fix3 = st.columns(3)
                with col_fix1:
                    st.metric("Load Demand", f"{base_cfg.total_load_mw:.0f} MW")
                    st.metric("Contracted Demand", f"{base_cfg.contracted_demand_mw:.0f} MW")
                with col_fix2:
                    st.metric("Evacuation Limit", f"{base_cfg.evac_limit_mw:.0f} MW")
                    st.metric("Project Lifetime", f"{st.session_state.get('project_lifetime_years_slider', 25)} years")
                with col_fix3:
                    st.metric("Solar Degradation", f"{base_cfg.solar_degrad*100:.2f}%/yr")
                    st.metric("BESS Efficiency", f"{base_cfg.one_way_eff*100:.1f}%")
                
                st.caption("💡 **Optimization searches for the cheapest RE system configuration to meet this fixed load demand**")
        else:
            st.warning("⚠️ **Run a simulation first** to set the fixed parameters for optimization")
            st.stop()
        
        st.markdown("---")
        
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.subheader("🔍 Search Space")
            
            # Solar range
            st.markdown("**☀️ Solar Capacity**")
            solar_opt_range = st.slider(
                "Range (MW AC)",
                min_value=0,
                max_value=2000,
                value=(600, 1200),
                step=50,
                key="opt_solar_range"
            )
            solar_opt_step = st.number_input("Step (MW)", min_value=50, max_value=500, value=100, key="opt_solar_step")
            
            # Wind range
            st.markdown("**🌬️ Wind Turbines**")
            wind_opt_range = st.slider(
                "WTG Count Range",
                min_value=0,
                max_value=150,
                value=(0, 100),
                step=10,
                key="opt_wind_range"
            )
            wind_opt_step = st.number_input("Step (WTG)", min_value=10, max_value=50, value=20, key="opt_wind_step")
            
            # BESS toggle and range
            st.markdown("**🔋 Battery Storage**")
            bess_opt_enabled = st.checkbox("Include BESS in optimization", value=True, key="opt_bess_enabled")
            
            if bess_opt_enabled:
                bess_power_opt_range = st.slider(
                    "Power Range (MW)",
                    min_value=0,
                    max_value=300,
                    value=(0, 200),
                    step=25,
                    key="opt_bess_power_range"
                )
                bess_power_opt_step = st.number_input("Power Step (MW)", min_value=25, max_value=100, value=50, key="opt_bess_power_step")
                
                bess_energy_opt_range = st.slider(
                    "Energy Range (MWh)",
                    min_value=0,
                    max_value=1000,
                    value=(0, 800),
                    step=50,
                    key="opt_bess_energy_range"
                )
                bess_energy_opt_step = st.number_input("Energy Step (MWh)", min_value=50, max_value=400, value=200, key="opt_bess_energy_step")
            else:
                bess_power_opt_range = (0, 0)
                bess_power_opt_step = 1
                bess_energy_opt_range = (0, 0)
                bess_energy_opt_step = 1
            
            # Constraints
            st.markdown("**⚙️ Performance Constraints**")
            min_captive = st.slider("Min Captive Supply (%)", 0, 100, 0, 5, key="opt_min_captive")
            max_grid_import = st.slider("Max Grid Import (%)", 0, 100, 100, 5, key="opt_max_grid_import")
            
            # Calculate number of configurations
            st.markdown("---")
            n_solar = int((solar_opt_range[1] - solar_opt_range[0]) / solar_opt_step) + 1
            n_wind = int((wind_opt_range[1] - wind_opt_range[0]) / wind_opt_step) + 1
            n_bess_p = int((bess_power_opt_range[1] - bess_power_opt_range[0]) / bess_power_opt_step) + 1 if bess_opt_enabled else 1
            n_bess_e = int((bess_energy_opt_range[1] - bess_energy_opt_range[0]) / bess_energy_opt_step) + 1 if bess_opt_enabled else 1
            total_configs = n_solar * n_wind * n_bess_p * n_bess_e
            
            st.success(f"**📊 Search Space:** {total_configs:,} configurations\n\n**⏱️ Est. Time:** {total_configs * 0.3 / 60:.1f} - {total_configs * 0.6 / 60:.1f} minutes")
            
            # Run button
            run_opt_button = st.button("🚀 Run Optimization", type="primary", use_container_width=True, disabled=st.session_state.optimization_running)
        
        with col_right:
            st.subheader("📊 Results")
            
            # Show results if available
            if st.session_state.optimization_results is not None:
                results_df = st.session_state.optimization_results
                
                # Calculate feasibility statistics
                n_feasible = results_df['is_feasible'].sum()
                n_infeasible = (~results_df['is_feasible']).sum()
                n_total = len(results_df)
                
                # Get best overall and best feasible
                best_overall = results_df.iloc[0]
                feasible_results = results_df[results_df['is_feasible']]
                
                if len(feasible_results) > 0:
                    best_feasible = feasible_results.iloc[0]
                    best_result = best_feasible
                    all_good = True
                else:
                    best_result = best_overall
                    all_good = False
                
                # Feasibility summary
                st.markdown("### 📊 Optimization Summary")
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("Total Evaluated", f"{n_total:,}")
                with col_s2:
                    st.metric("✅ Feasible", f"{n_feasible:,}", delta=f"{n_feasible/n_total*100:.1f}%" if n_total > 0 else "0%")
                with col_s3:
                    st.metric("❌ Infeasible", f"{n_infeasible:,}", delta=f"{n_infeasible/n_total*100:.1f}%" if n_total > 0 else "0%", delta_color="inverse")
                
                # Warning if no feasible solutions
                if not all_good:
                    st.error("⚠️ **No feasible solutions found!** Showing best result but it violates constraints.")
                    if len(best_result.get('constraint_violations', [])) > 0:
                        st.warning(f"**Violations:** {', '.join(best_result['constraint_violations'])}")
                    st.info("💡 **Suggestions:** Relax constraints, expand search range, or increase system capacity.")
                else:
                    st.success(f"✅ Found {n_feasible:,} feasible solution(s)!")
                
                st.markdown("---")
                
                with st.container():
                    st.markdown("### 🏆 Best Configuration")
                    
                    # Metrics in columns
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("LCOE", f"₹{best_result['lcoe_inr_per_kwh']:.4f}/kWh")
                    with m2:
                        st.metric("Project Cost", f"₹{best_result['project_cost_crore']:.0f} Cr")
                    with m3:
                        st.metric("Annual OPEX", f"₹{best_result['annual_opex_crore']:.1f} Cr")
                    
                    # Configuration details
                    st.markdown("**System Configuration:**")
                    st.write(f"- Solar: **{best_result['solar_ac_mw']:.0f} MW**")
                    st.write(f"- Wind: **{best_result['wind_wtg_count']:.0f} WTG** ({best_result['wind_mw']:.0f} MW)")
                    st.write(f"- BESS: **{best_result['bess_power_mw']:.0f} MW / {best_result['bess_energy_mwh']:.0f} MWh**")
                    st.write(f"- Total Capacity: **{best_result['solar_ac_mw'] + best_result['wind_mw']:.0f} MW**")
                    
                    st.markdown("**Performance:**")
                    captive_icon = "✅" if best_result['is_feasible'] else "❌"
                    st.write(f"- Captive Supply: **{best_result['captive_percent']:.1f}%** {captive_icon}")
                    st.write(f"- Grid Import: **{best_result['grid_import_percent']:.1f}%**")
                    st.write(f"- CUF: **{best_result['cuf_percent']:.1f}%**")
                    st.write(f"- Curtailment: **{best_result['curtailment_gwh']:.1f} GWh/yr**")
                    
                    # Show feasibility status
                    if best_result['is_feasible']:
                        st.success("✅ **Status: FEASIBLE** - Meets all constraints")
                    else:
                        st.error("❌ **Status: INFEASIBLE** - Violates constraints")
                        if len(best_result.get('constraint_violations', [])) > 0:
                            for violation in best_result['constraint_violations']:
                                st.caption(f"  • {violation}")
                
                # Action buttons
                st.markdown("---")
                col_a, col_b = st.columns(2)
                with col_a:
                    # Download best config as JSON
                    best_config_json = {
                        "solar_ac_mw": float(best_result['solar_ac_mw']),
                        "wind_wtg_count": int(best_result['wind_wtg_count']),
                        "bess_power_mw": float(best_result['bess_power_mw']),
                        "bess_energy_mwh": float(best_result['bess_energy_mwh'])
                    }
                    st.download_button(
                        "📥 Download Best Config",
                        data=json.dumps(best_config_json, indent=2),
                        file_name="best_config.json",
                        mime="application/json",
                        use_container_width=True
                    )
                with col_b:
                    # Download all results CSV
                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        "📊 Download All Results",
                        data=csv_buffer.getvalue(),
                        file_name="optimization_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                # Top 10 results table with tabs for all vs feasible
                st.markdown("### 📈 Top Configurations")
                
                tab_all, tab_feasible = st.tabs(["All Results", f"Feasible Only ({n_feasible})"])
                
                with tab_all:
                    top_10_all = results_df.head(10)[['solar_ac_mw', 'wind_wtg_count', 'bess_power_mw', 'bess_energy_mwh', 
                                                   'lcoe_inr_per_kwh', 'captive_percent', 'grid_import_percent', 'is_feasible']].copy()
                    top_10_all.columns = ['Solar (MW)', 'Wind (WTG)', 'BESS Power (MW)', 'BESS Energy (MWh)', 
                                      'LCOE (₹/kWh)', 'Captive %', 'Grid %', 'Feasible']
                    
                    # Add visual indicator
                    def highlight_feasible(row):
                        if row['Feasible']:
                            return ['background-color: #d4edda'] * len(row)
                        else:
                            return ['background-color: #f8d7da'] * len(row)
                    
                    styled_df = top_10_all.style.apply(highlight_feasible, axis=1)
                    st.dataframe(styled_df, use_container_width=True, height=400)
                    st.caption("🟢 Green = Feasible | 🔴 Red = Infeasible (violates constraints)")
                
                with tab_feasible:
                    if len(feasible_results) > 0:
                        top_10_feasible = feasible_results.head(10)[['solar_ac_mw', 'wind_wtg_count', 'bess_power_mw', 'bess_energy_mwh', 
                                                       'lcoe_inr_per_kwh', 'captive_percent', 'grid_import_percent']].copy()
                        top_10_feasible.columns = ['Solar (MW)', 'Wind (WTG)', 'BESS Power (MW)', 'BESS Energy (MWh)', 
                                              'LCOE (₹/kWh)', 'Captive %', 'Grid %']
                        st.dataframe(top_10_feasible, use_container_width=True, height=400)
                    else:
                        st.warning("No feasible configurations found. Adjust constraints or search space.")
                
            elif st.session_state.optimization_running:
                st.info("⏳ Optimization in progress... This may take several minutes.")
            else:
                st.info("Configure search space and click 'Run Optimization' to find the best configuration")
        
        # Run optimization when button clicked
        if run_opt_button:
            if st.session_state.cfg is None or st.session_state.df_profiles is None:
                st.error("❌ Please run a simulation first before optimizing!")
                st.stop()
                
            st.session_state.optimization_running = True
            
            # Get base configuration from session state
            base_cfg = st.session_state.cfg
            
            # Build optimization config
            opt_config = OptimizationConfig(
                solar_ac_mw_min=float(solar_opt_range[0]),
                solar_ac_mw_max=float(solar_opt_range[1]),
                solar_ac_mw_step=float(solar_opt_step),
                wind_wtg_count_min=int(wind_opt_range[0]),
                wind_wtg_count_max=int(wind_opt_range[1]),
                wind_wtg_count_step=int(wind_opt_step),
                wind_wtg_capacity_mw=3.3,
                bess_power_mw_min=float(bess_power_opt_range[0]),
                bess_power_mw_max=float(bess_power_opt_range[1]),
                bess_power_mw_step=float(bess_power_opt_step),
                bess_energy_mwh_min=float(bess_energy_opt_range[0]),
                bess_energy_mwh_max=float(bess_energy_opt_range[1]),
                bess_energy_mwh_step=float(bess_energy_opt_step),
                max_evacuation_limit_mw=base_cfg.evac_limit_mw,
                min_captive_percent=float(min_captive),
                max_grid_import_percent=float(max_grid_import),
                bess_mode=base_cfg.bess_mode
            )
            
            # Generate configurations
            with st.spinner("Generating configurations..."):
                configurations = generate_configurations(opt_config)
            
            st.info(f"Evaluating {len(configurations)} configurations...")
            
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Load financial parameters
            with open(st.session_state.financial_file) as f:
                fin_data = json.load(f)
            
            # Cost and financial params
            cost_params = CostParameters(residual_value_fraction=0.0)
            fin_params = FinancialParameters(
                project_lifetime_years=st.session_state.project_lifetime_years_slider,
                system_degradation=base_cfg.solar_degrad,
                discount_rate=fin_data.get('discount_rate', 0.0749997),
                tax_rate=fin_data.get('tax_rate', 0.2782),
                interest_rate=fin_data.get('loan_interest_rate', 0.095),
                opex_escalation=0.05,
                equity_fraction=fin_data.get('equity_percent', 0.3),
                loan_fraction=fin_data.get('loan_percent', 0.7),
                loan_term_years=int(fin_data.get('loan_term_years', 10))
            )
            
            # Base config from current simulation
            base_config = base_cfg
            df_profiles = st.session_state.df_profiles
            
            # Define evaluation function that uses stored profiles
            def evaluate_config_streamlit(solar, wind, bess_p, bess_e):
                """Evaluate a configuration using stored profile data."""
                from banking_settlement_skeleton import SettlementConfig, settle
                from lcoe_calculator import calculate_project_cost, calculate_annual_opex
                
                # Create config for this evaluation
                eval_cfg = Config(
                    year=base_config.year,
                    total_load_mw=base_config.total_load_mw,
                    existing_solar_mwp=base_config.existing_solar_mwp,
                    contracted_demand_mw=base_config.contracted_demand_mw,
                    evac_limit_mw=opt_config.max_evacuation_limit_mw,
                    tl_loss=base_config.tl_loss,
                    wheeling_loss=base_config.wheeling_loss,
                    dc_ac_ratio=base_config.dc_ac_ratio,
                    solar_ac_mw=solar,
                    wind_wtg_count=wind,
                    wind_expected_cuf=base_config.wind_expected_cuf,
                    wind_reference_cuf=base_config.wind_reference_cuf,
                    solar_degrad=base_config.solar_degrad,
                    wind_degrad=base_config.wind_degrad,
                    p_multiplier=base_config.p_multiplier,
                    banking_enabled=base_config.banking_enabled,
                    bess_mode=opt_config.bess_mode,
                    one_way_eff=base_config.one_way_eff,
                    bess_power_mw=bess_p,
                    bess_energy_mwh=bess_e,
                    soc_start_gwh=base_config.soc_start_gwh,
                )
                
                # Run hourly simulation
                df_hourly = simulate_hourly(eval_cfg, df_profiles)
                
                # Run settlement
                scfg = SettlementConfig()
                settlement_results = settle(df_hourly, scfg)
                
                # Calculate metrics
                annual_generation = df_hourly["gen_total"].sum() / 1e6  # GWh
                annual_load = df_hourly["net_load"].sum() / 1e6  # GWh
                captive_supply = (df_hourly["delivered_total"].sum() + settlement_results["annual_summary"]["total_banked_used"]) / 1e6
                grid_import = df_hourly["from_grid"].sum() / 1e6
                curtailment = df_hourly["curtailed"].sum() / 1e6
                
                captive_percent = (captive_supply / annual_load * 100) if annual_load > 0 else 0
                grid_import_percent = (grid_import / annual_load * 100) if annual_load > 0 else 0
                
                wind_mw = wind * 3.3
                total_capacity = solar + wind_mw
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
                project_cost = calculate_project_cost(solar, wind_mw, bess_p, bess_e, cost_params)
                annual_opex = calculate_annual_opex(solar, wind_mw, bess_e, cost_params)
                
                # Energy delivered over project lifetime
                first_year_energy_kwh = captive_supply * 1e9  # GWh to kWh
                annual_energy_array = np.full(fin_params.project_lifetime_years, first_year_energy_kwh)
                
                # Calculate LCOE
                from lcoe_calculator import calculate_lcoe as calc_lcoe
                lcoe_results = calc_lcoe(project_cost, annual_opex, annual_energy_array, fin_params, cost_params)
                
                return {
                    'solar_ac_mw': solar,
                    'wind_wtg_count': int(wind),
                    'wind_mw': wind_mw,
                    'bess_power_mw': bess_p,
                    'bess_energy_mwh': bess_e,
                    'lcoe_inr_per_kwh': lcoe_results['lcoe_inr_per_kwh'],
                    'project_cost_crore': project_cost,
                    'annual_opex_crore': annual_opex,
                    'annual_generation_gwh': annual_generation,
                    'annual_load_gwh': annual_load,
                    'captive_supply_gwh': captive_supply,
                    'grid_import_gwh': grid_import,
                    'curtailment_gwh': curtailment,
                    'captive_percent': captive_percent,
                    'grid_import_percent': grid_import_percent,
                    'cuf_percent': cuf_percent,
                    'is_feasible': is_feasible,
                    'constraint_violations': violations,
                }
            
            # Run evaluations
            results = []
            start_time = time.time()
            n_feasible_so_far = 0
            
            for i, (solar, wind, bess_p, bess_e) in enumerate(configurations):
                try:
                    # Evaluate using stored profiles
                    result = evaluate_config_streamlit(solar, wind, bess_p, bess_e)
                    results.append(result)
                    
                    if result['is_feasible']:
                        n_feasible_so_far += 1
                    
                    # Update progress
                    progress = (i + 1) / len(configurations)
                    progress_bar.progress(progress)
                    
                    if (i + 1) % 10 == 0:
                        elapsed = time.time() - start_time
                        remaining = (elapsed / (i + 1)) * (len(configurations) - i - 1)
                        status_text.text(f"Evaluated {i+1}/{len(configurations)} | Feasible: {n_feasible_so_far} | Elapsed: {elapsed/60:.1f}m | Remaining: {remaining/60:.1f}m")
                except Exception as e:
                    st.warning(f"Config {i+1} failed: {e}")
                    continue
            
            # Convert to DataFrame and sort by LCOE
            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values('lcoe_inr_per_kwh').reset_index(drop=True)
            
            # Calculate summary statistics
            n_feasible_final = df_results['is_feasible'].sum()
            n_total_final = len(df_results)
            
            # Store in session state
            st.session_state.optimization_results = df_results
            st.session_state.optimization_running = False
            
            elapsed_total = time.time() - start_time
            
            # Show completion message with statistics
            if n_feasible_final > 0:
                st.success(f"✅ Optimization complete in {elapsed_total/60:.1f} minutes! Found {n_feasible_final}/{n_total_final} feasible solutions.")
            else:
                st.warning(f"⚠️ Optimization complete in {elapsed_total/60:.1f} minutes. No feasible solutions found among {n_total_final} configurations.")
            
            st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Hybrid RE Optimizer v1.0**")
st.sidebar.markdown("Built with Streamlit")
