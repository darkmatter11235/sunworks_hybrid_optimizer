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

from hourly_sim_skeleton import Config, simulate_hourly, assign_tod
from lcoe_calculator import calculate_lcoe

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
        config_file = demo_path / "system_config.json"
        profiles_file = demo_path / "generation_profiles.csv"
        load_file = demo_path / "load_profile.csv"
        financial_file = demo_path / "financial_params.json"
        files_exist = all([f.exists() for f in [config_file, profiles_file, load_file, financial_file]])
    
    if not files_exist:
        st.warning("⚠️ Demo data not available. Please upload an Excel file or use local files.")
        st.sidebar.markdown("**Deployment Note**: Include `standalone_data/` in your repository for demo mode.")
        st.stop()
    
    st.session_state.data_loaded = True

elif data_source == "Upload Excel File":
    st.sidebar.markdown("Upload your Excel file to extract data")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose Excel file", 
        type=['xlsx', 'xls'],
        help="Upload your hybrid RE system Excel model"
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
                profiles_file = data_path / "generation_profiles.csv"
                load_file = data_path / "load_profile.csv"
                financial_file = data_path / "financial_params.json"
                
                files_exist = True
                st.sidebar.success("✅ Data extracted successfully!")
                st.session_state.data_loaded = True
                
            except Exception as e:
                st.sidebar.error(f"Error extracting data: {e}")
                st.stop()
    else:
        st.info("👈 Upload an Excel file to get started")
        st.markdown("""
        ### How to use:
        1. Upload your hybrid RE Excel model
        2. Configure system parameters (solar, wind, BESS)
        3. Click "Run Simulation" to see results
        
        **Don't have an Excel file?** Switch to "Use Demo Data" to try it out!
        """)
        st.stop()

else:  # Use Local Files
    data_dir = st.sidebar.text_input("Data Directory", "standalone_data")
    data_path = Path(data_dir)
    config_file = data_path / "system_config.json"
    profiles_file = data_path / "generation_profiles.csv"
    load_file = data_path / "load_profile.csv"
    financial_file = data_path / "financial_params.json"
    
    files_exist = all([f.exists() for f in [config_file, profiles_file, load_file, financial_file]])
    
    if not files_exist:
        st.sidebar.error("⚠️ Data files not found!")
        st.sidebar.markdown(f"""
        Required files in `{data_dir}`:
        - system_config.json
        - generation_profiles.csv
        - load_profile.csv
        - financial_params.json
        
        Run: `python utils/extract_standalone_data.py your_file.xlsx`
        """)
        st.stop()
    else:
        st.sidebar.success("✓ Data files loaded")
        st.session_state.data_loaded = True

# Load default configuration
with open(config_file) as f:
    default_config = json.load(f)

# Main tabs
tab1, tab2, tab3 = st.tabs(["🔧 Configuration", "📊 Simulation Results", "🎯 Optimization"])

with tab1:
    st.header("System Configuration")
    
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
    
    # Run simulation button
    if st.button("▶️ Run Simulation", type="primary", use_container_width=True):
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
            }
            filtered_default.update(user_inputs)
            
            cfg = Config(**filtered_default)
            
            # Load profiles
            df_gen = pd.read_csv(profiles_file)
            df_load = pd.read_csv(load_file)
            
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
            
            # Run simulation
            df_sim = simulate_hourly(cfg, df_profiles)
            
            # Calculate LCOE
            try:
                lcoe = calculate_lcoe(cfg, df_sim, str(financial_file))
            except Exception as e:
                lcoe = None
                st.warning(f"Could not calculate LCOE: {e}")
            
            # Store in session
            st.session_state.df_sim = df_sim
            st.session_state.cfg = cfg
            st.session_state.lcoe = lcoe
            st.session_state.simulation_run = True
            
        st.success("✅ Simulation complete!")
        st.rerun()

with tab2:
    if not st.session_state.simulation_run:
        st.info("👈 Configure system and run simulation to see results")
    else:
        df_sim = st.session_state.df_sim
        cfg = st.session_state.cfg
        lcoe = st.session_state.lcoe
        
        st.header("Simulation Results")
        
        # Key metrics
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
                st.metric("LCOE", f"₹{lcoe:.4f}/kWh")
            else:
                st.metric("LCOE", "N/A")
        
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
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
        
        # Charts
        st.subheader("📈 Visualizations")
        
        # Sample week
        week_start = 24 * 7 * 20  # Week 20
        week_end = week_start + 24 * 7
        df_week = df_sim.iloc[week_start:week_end]
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df_week.index, y=df_week['solar_gen'], name='Solar', fill='tozeroy'))
        fig1.add_trace(go.Scatter(x=df_week.index, y=df_week['wind_gen'], name='Wind', fill='tozeroy'))
        fig1.add_trace(go.Scatter(x=df_week.index, y=df_week['load_total'], name='Load', line=dict(color='red', dash='dash')))
        fig1.update_layout(title="Generation vs Load (Sample Week)", xaxis_title="Hour", yaxis_title="MW", hovermode='x unified')
        st.plotly_chart(fig1, use_container_width=True)
        
        # BESS SOC
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_week.index, y=df_week['soc_mwh'], name='SOC', fill='tozeroy'))
        fig2.update_layout(title="BESS State of Charge (Sample Week)", xaxis_title="Hour", yaxis_title="MWh", hovermode='x')
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.header("🎯 Optimization")
    st.markdown("Find the best configuration to minimize LCOE")
    
    st.subheader("Search Space")
    col1, col2 = st.columns(2)
    
    with col1:
        solar_range = st.slider(
            "Solar Capacity Range (MW)",
            min_value=0,
            max_value=2000,
            value=(500, 1000),
            step=50
        )
        solar_step = st.number_input("Solar Step (MW)", min_value=10, max_value=200, value=50)
    
    with col2:
        wind_range = st.slider(
            "Wind WTG Count Range",
            min_value=0,
            max_value=100,
            value=(0, 20),
            step=5
        )
        wind_step = st.number_input("Wind Step", min_value=1, max_value=10, value=5)
    
    bess_enabled = st.checkbox("Include BESS in optimization", value=True)
    
    if bess_enabled:
        col3, col4 = st.columns(2)
        with col3:
            bess_power_range = st.slider(
                "BESS Power Range (MW)",
                min_value=0,
                max_value=200,
                value=(25, 75),
                step=5
            )
        with col4:
            bess_energy_range = st.slider(
                "BESS Energy Range (MWh)",
                min_value=0,
                max_value=500,
                value=(100, 300),
                step=25
            )
    
    max_configs = st.number_input(
        "Max Configurations to Evaluate",
        min_value=10,
        max_value=10000,
        value=100,
        help="More configs = better optimization but slower"
    )
    
    if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
        st.warning("⚠️ Optimization feature coming soon! Use optimizer.py for now.")
        st.code(f"""
# Run optimization from command line:
python optimizer.py \\
    --solar-min {solar_range[0]} --solar-max {solar_range[1]} --solar-step {solar_step} \\
    --wind-min {wind_range[0]} --wind-max {wind_range[1]} --wind-step {wind_step} \\
    --output optimization_results.csv
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Hybrid RE Optimizer v1.0**")
st.sidebar.markdown("Built with Streamlit")
