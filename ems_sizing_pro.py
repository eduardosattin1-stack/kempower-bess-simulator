import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import io, os, json
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4

# --- PATH & BRANDING ---
SCRIPT_DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()
LOGO_PATH = SCRIPT_DIR / "kempowerlogo.png"
KEMPOWER_ORANGE = "#FF6400"

st.set_page_config(page_title="Kempower | BESS Sizing", layout="wide")

st.markdown(f"""
    <style>
    .reportview-container .main .block-container {{ padding-top: 1rem; }}
    .stMetric {{ background-color: #ffffff; padding: 10px; border-radius: 8px; border-left: 5px solid {KEMPOWER_ORANGE}; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
    .kpi-header {{ font-size: 1.1rem; font-weight: bold; margin-bottom: 10px; color: #333; }}
    </style>
    """, unsafe_allow_html=True)

# --- CONFIGURATION MAPS ---
# Defined BESS Power Ratings (kWh -> kW)
BESS_POWER_MAP = {
    140: 75,
    280: 150,
    420: 225,
    560: 300,
    840: 450,
    1120: 600,
    1960: 1050,
    2240: 1200,
    3360: 1800,
    4480: 2400,
    5600: 3000
}

# --- SIDEBAR ---
with st.sidebar:
    if os.path.exists(LOGO_PATH): st.image(str(LOGO_PATH), width=180)
    st.markdown("### 🏢 Site Parameters")
    grid_limit = st.number_input("Grid Limit (kW)", value=150)
    num_plugs = st.number_input("Plugs", value=10)
    charger_cap = st.number_input("Site Charger Capacity (kW)", value=600, help="Physical limit of charging hardware")
    
    cap_options = sorted(list(BESS_POWER_MAP.keys()))
    nominal_cap = st.selectbox("BESS Capacity (kWh)", cap_options, index=cap_options.index(560))
    
    bess_max_power = BESS_POWER_MAP[nominal_cap]
    st.sidebar.caption(f"ℹ️ BESS Power Rating: **{bess_max_power} kW**")
    
    usable_factor = st.slider("Usable Factor", 0.5, 1.0, 0.85)
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload 7-Day CSV Profile", type="csv")
    load_multiplier = st.slider("Scaling Factor", 0.5, 3.0, 1.0)
    growth_rate = st.slider("Annual Growth (%)", 0, 20, 5) / 100

if uploaded_file:
    # --- DATA LOADING ---
    try:
        df_base = pd.read_csv(uploaded_file, sep=None, engine='python')
    except:
        uploaded_file.seek(0)
        df_base = pd.read_csv(uploaded_file, sep=';')
        
    if df_base.shape[1] < 2: 
        uploaded_file.seek(0)
        df_base = pd.read_csv(uploaded_file, sep=';')

    df_base.columns = df_base.columns.str.strip()
    
    if 'Timestamp' in df_base.columns: df_base.rename(columns={'Timestamp': 'timestamp'}, inplace=True)
    if 'timestamp' not in df_base.columns: df_base.rename(columns={df_base.columns[0]: 'timestamp'}, inplace=True)
    
    if 'load_kw' not in df_base.columns:
         if 'Power [kW]' in df_base.columns:
             df_base.rename(columns={'Power [kW]': 'raw_load'}, inplace=True)
         else:
             df_base.rename(columns={df_base.columns[1]: 'raw_load'}, inplace=True)
    else:
        df_base.rename(columns={'load_kw': 'raw_load'}, inplace=True)

    df_base['timestamp'] = pd.to_datetime(df_base['timestamp'], dayfirst=True)
    df_base['day_name'] = df_base['timestamp'].dt.day_name()
    
    # --- SIMULATION PARAMETERS ---
    step_hrs, year_mult = 5/60, 365/7
    current_usable_kwh = nominal_cap * usable_factor
    initial_kwh = current_usable_kwh
    total_lifetime_cycles, total_missed_mwh, total_ev_mwh = 0, 0, 0
    yearly_results, plot_data = [], {}

    first_qos_warning_idx = -1

    for year in range(1, 11):
        df = df_base.copy()
        
        # 1. Theoretical Demand (Uncapped growth)
        df['load_theoretical'] = df['raw_load'] * load_multiplier * ((1 + growth_rate) ** (year - 1))
        
        # 2. Physical Load to Serve (Capped by Charger Capacity)
        # The Grid+BESS system can only attempt to serve what the chargers can output.
        df['load_to_serve'] = np.minimum(df['load_theoretical'], charger_cap)
        
        n = len(df)
        grid_used, bess_disc, bess_char = np.zeros(n), np.zeros(n), np.zeros(n)
        missed_energy_yr, current_soc = 0, current_usable_kwh * 0.5

        for i in range(n):
            # We try to meet the 'load_to_serve'
            demand = df['load_to_serve'].iloc[i]
            max_p = bess_max_power 
            
            if demand <= grid_limit:
                grid_used[i] = demand
                can_acc = ((current_usable_kwh - current_soc) / step_hrs) / 0.85
                bess_char[i] = min(grid_limit - demand, can_acc, max_p)
            else:
                grid_used[i] = grid_limit
                can_prov = (current_soc / step_hrs) * 0.85
                bess_disc[i] = min(demand - grid_limit, can_prov, max_p)
            
            current_soc += (bess_char[i] * 0.85 - bess_disc[i] / 0.85) * step_hrs
            
            # Missed Energy Logic
            # Missed = What the EVs wanted (Theoretical) - What we delivered (Grid + BESS)
            delivered = grid_used[i] + bess_disc[i]
            unmet = df['load_theoretical'].iloc[i] - delivered
            if unmet > 0.001:
                missed_energy_yr += unmet * step_hrs

        # Aggregations
        yr_ev_mwh = (df['load_theoretical'].sum() * step_hrs * year_mult) / 1000
        yr_missed_mwh = (missed_energy_yr * year_mult) / 1000
        
        total_missed_mwh += yr_missed_mwh
        total_ev_mwh += yr_ev_mwh
        
        yr_bess_mwh = (bess_disc.sum() * step_hrs * year_mult) / 1000
        yearly_cycles = (yr_bess_mwh * 1000) / current_usable_kwh
        total_lifetime_cycles += yearly_cycles
        
        soh_pct = (current_usable_kwh/initial_kwh)*100
        qos_val = round((yr_missed_mwh / yr_ev_mwh) * 100, 2)
        
        soh_label = f"{soh_pct:.1f}%" + (" 🚩" if soh_pct < 70 else "")
        if first_qos_warning_idx == -1 and qos_val > 5:
            first_qos_warning_idx = year - 1
        
        yearly_results.append({
            "Year": year, "SoH (%)": soh_label, "Cap (kWh)": round(current_usable_kwh, 1),
            "Cycles": int(yearly_cycles), "EV Load (MWh)": round(yr_ev_mwh, 1),
            "Util (%)": round((df['load_to_serve'].mean() / charger_cap) * 100, 1),
            "QoS Loss (%)": qos_val, "Plug Thr": round((yr_bess_mwh * 1000) / (365 * num_plugs), 1)
        })

        df['grid_used'], df['bess_disc'] = grid_used, bess_disc
        plot_data[year] = df.copy()
        current_usable_kwh *= 0.985
        for _ in range(int(yearly_cycles)): 
            rate = 0.0045 if total_lifetime_cycles <= 800 else (0.0018 if total_lifetime_cycles <= 2500 else 0.0055)
            current_usable_kwh *= (1 - (rate / 100))

    if first_qos_warning_idx != -1:
        yearly_results[first_qos_warning_idx]["QoS Loss (%)"] = f"{yearly_results[first_qos_warning_idx]['QoS Loss (%)']} ⚠️"

    # --- TOP KPIs ---
    st.markdown('<p class="kpi-header">KPIs after 10 years of operation</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cycles", int(total_lifetime_cycles))
    c2.metric("Final BESS SoH", f"{(current_usable_kwh/initial_kwh)*100:.1f}%")
    c3.metric("Unmet Demand Ratio", f"{(total_missed_mwh/total_ev_mwh)*100:.2f}%")
    c4.metric("Avg Plug Throughput", f"{np.mean([x['Plug Thr'] for x in yearly_results]):.1f} kWh/d")

    # --- CHART ---
    st.write("---")
    view_yr = st.selectbox("Year Profile:", list(range(1, 11)), label_visibility="collapsed")
    p_df = plot_data[view_yr]
    
    day_indices, unique_days = [], []
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        indices = p_df.index[p_df['day_name'] == day].tolist()
        if indices: day_indices.append(indices[len(indices)//2]); unique_days.append(day)

    fig = go.Figure()
    # Grid Power
    fig.add_trace(go.Bar(name='Grid', x=p_df.index, y=p_df['grid_used'], marker_color='#2b579a', opacity=0.8, hovertemplate='%{y} kW<extra></extra>'))
    # BESS Power
    fig.add_trace(go.Bar(name='BESS Supplement', x=p_df.index, y=p_df['bess_disc'], marker_color='#7eb26d', hovertemplate='%{y} kW<extra></extra>'))
    
    # EV Demand Line (Using Theoretical Load to show potential undersizing)
    fig.add_trace(go.Scatter(name='EV Demand (Theoretical)', x=p_df.index, y=p_df['load_theoretical'], 
                             line=dict(color=KEMPOWER_ORANGE, width=2, dash='dot'), hovertemplate='%{y} kW<extra></extra>'))
    
    # Charger Capacity Limit Line
    fig.add_shape(type="line", x0=p_df.index[0], x1=p_df.index[-1], y0=charger_cap, y1=charger_cap,
                  line=dict(color="Red", width=1, dash="dashdot"))
    
    fig.update_layout(
        barmode='stack', 
        margin=dict(l=0, r=0, t=50, b=30), 
        height=450, 
        yaxis_title="Power (kW)", 
        xaxis=dict(tickmode='array', tickvals=day_indices, ticktext=unique_days), 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- EXPORTS ---
    st.write("---")
    res_col, qos_col = st.columns([2.2, 1])
    with res_col:
        st.markdown("**10-Year Study Summary**")
        final_df = pd.DataFrame(yearly_results).set_index("Year")
        st.dataframe(final_df, use_container_width=True, height=385)

    with qos_col:
        st.markdown("**Quality of Service (QoS) Legend**")
        st.markdown('<div class="qos-text"><b>0% - 1% Optimal:</b> High reliability infrastructure.</div>', unsafe_allow_html=True)
        st.markdown('<div class="qos-text"><b>2% - 5% Warning:</b> Risk of throttled user sessions.</div>', unsafe_allow_html=True)
        st.markdown('<div class="qos-text"><b>> 5% Critical:</b> Unmet demand exceeds acceptable thresholds.</div>', unsafe_allow_html=True)
        
        def generate_pdf(res_df):
            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=A4)
            els = []
            styles = getSampleStyleSheet()
            if os.path.exists(LOGO_PATH): els.append(RLImage(str(LOGO_PATH), width=120, height=40)); els.append(Spacer(1, 10))
            els.append(Paragraph("Kempower | BESS Sizing Report", styles['Title']))
            els.append(Spacer(1, 15))
            data = [["Year"] + res_df.columns.tolist()] + [[i] + row for i, row in zip(res_df.index, res_df.values.tolist())]
            t = Table(data)
            t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor(KEMPOWER_ORANGE)), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('FONTSIZE', (0,0), (-1,-1), 8)]))
            els.append(t)
            doc.build(els)
            return buf.getvalue()
            
        def generate_smart_export(base_df, metadata):
            buffer = io.StringIO()
            json_meta = json.dumps(metadata)
            buffer.write(f"# METADATA_JSON:{json_meta}\n")
            
            # Export the RAW load so the Business App can re-apply its own scaling/capping logic
            export_df = base_df[['timestamp', 'raw_load']].copy()
            export_df.columns = ['timestamp', 'load_kw'] 
            export_df['timestamp'] = export_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            export_df.to_csv(buffer, index=False)
            return buffer.getvalue().encode('utf-8')

        st.divider()
        st.download_button("📥 Results PDF Report", generate_pdf(final_df), "Kempower_BESS_Report.pdf", "application/pdf", use_container_width=True)
        
        final_soh_val = (current_usable_kwh/initial_kwh)*100
        approx_deg_rate = (100 - final_soh_val) / 10
        avg_plug_val = np.mean([x['Plug Thr'] for x in yearly_results])

        meta = {
            'grid_limit': grid_limit,
            'bess_capacity': nominal_cap,
            'charger_cap': charger_cap,
            'growth_rate': growth_rate * 100,
            'degradation_rate': approx_deg_rate,
            'load_multiplier': load_multiplier,
            'total_cycles': int(total_lifetime_cycles),
            'final_soh': round(final_soh_val, 1),
            'avg_plug_thru': round(avg_plug_val, 1)
        }
        
        csv_data = generate_smart_export(df_base, meta)
        st.download_button("📊 Export Data for Trade-Off", csv_data, "kempower_smart_export.csv", "text/csv", use_container_width=True)
        
else:
    st.title("⚡ Kempower | BESS Sizing Tool v5.6")
    st.info("Please upload a 7-day ChargEye load profile to begin the simulation.")