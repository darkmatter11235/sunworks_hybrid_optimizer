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
        
        # Project Summary
        st.subheader(f"📅 Project: {project_years} Years | Lifetime Energy: {total_lifetime_gwh:,.1f} GWh")
        
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
        
        # Charts
        st.subheader("📈 Visualizations")
        
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
    
    if st.button("🚀 Run Optimization", type="primary", width='stretch'):
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
