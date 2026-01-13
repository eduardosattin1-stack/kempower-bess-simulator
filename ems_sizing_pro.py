import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import timedelta
import io
import os
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4

# --- PATH & BRANDING ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(SCRIPT_DIR, "kempowerlogo.png")
KEMPOWER_ORANGE = "#FF6400"

st.set_page_config(page_title="Kempower | BESS Sizing", layout="wide")

# CSS for a professional, no-scroll interface
st.markdown(f"""
    <style>
    .reportview-container .main .block-container {{ padding-top: 1rem; padding-bottom: 0rem; }}
    .stMetric {{ background-color: #ffffff; padding: 10px; border-radius: 8px; border-left: 5px solid {KEMPOWER_ORANGE}; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
    .qos-text {{ font-size: 0.82rem; color: #444; line-height: 1.3; margin-bottom: 8px; border-bottom: 1px solid #eee; padding-bottom: 4px; }}
    [data-testid="stDataFrame"] {{ border: none !important; }}
    .kpi-header {{ font-size: 1.1rem; font-weight: bold; margin-bottom: 10px; color: #333; }}
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    if os.path.exists(LOGO_PATH): st.image(LOGO_PATH, width=180)
    st.markdown("### 🏢 Site Parameters")
    grid_limit = st.number_input("Grid Limit (kW)", value=150)
    num_plugs = st.number_input("Plugs", value=10)
    charger_cap = st.number_input("Total Charger Power (kW)", value=600)
    nominal_cap = st.selectbox("BESS Capacity (kWh)", [280, 420, 560, 840, 1120, 1960, 2240, 3360, 4480, 5600], index=2)
    usable_factor = st.slider("Usable Factor", 0.5, 1.0, 0.85)
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Load Profile", type="csv")
    load_multiplier = st.slider("Scaling Factor", 0.5, 3.0, 1.0)
    growth_rate = st.slider("Annual Growth (%)", 0, 20, 5) / 100

if uploaded_file:
    # --- SIMULATION ENGINE ---
    df_base = pd.read_csv(uploaded_file, sep=None, engine='python')
    if df_base.shape[1] < 2: df_base = pd.read_csv(uploaded_file, sep=';')
    df_base.columns = ['timestamp', 'raw_load'] + list(df_base.columns[2:])
    df_base['timestamp'] = pd.to_datetime(df_base['timestamp'], dayfirst=True)
    df_base['day_name'] = df_base['timestamp'].dt.day_name()
    
    step_hrs, year_mult = 5/60, 365/7
    current_usable_kwh, total_lifetime_cycles, total_eff_loss_mwh = nominal_cap * usable_factor, 0, 0
    initial_kwh = current_usable_kwh
    yearly_results, plot_data = [], {}

    for year in range(1, 11):
        df = df_base.copy()
        df['load_kw'] = df['raw_load'] * load_multiplier * ((1 + growth_rate) ** (year - 1))
        n = len(df)
        grid_used, bess_disc, bess_char = np.zeros(n), np.zeros(n), np.zeros(n)
        missed_energy, current_soc, yr_loss_kwh = 0, current_usable_kwh * 0.5, 0

        for i in range(n):
            load, max_p = df['load_kw'].iloc[i], current_usable_kwh * 0.5
            if load <= grid_limit:
                grid_used[i] = load
                spare = grid_limit - load
                can_acc = ((current_usable_kwh - current_soc) / step_hrs) / 0.85
                bess_char[i] = min(spare, can_acc, max_p)
            else:
                grid_used[i] = grid_limit
                can_prov = (current_soc / step_hrs) * 0.85
                bess_disc[i] = min(load - grid_limit, can_prov, max_p)
                missed_energy += (load - grid_limit - bess_disc[i]) * step_hrs
            
            int_draw, int_gain = bess_disc[i] / 0.85, bess_char[i] * 0.85
            yr_loss_kwh += ((bess_char[i] - int_gain) + (int_draw - bess_disc[i])) * step_hrs
            current_soc += (int_gain - int_draw) * step_hrs

        yr_ev_mwh = (df['load_kw'].sum() * step_hrs * year_mult) / 1000
        yr_bess_mwh = (bess_disc.sum() * step_hrs * year_mult) / 1000
        yr_loss_mwh = (yr_loss_kwh * year_mult) / 1000
        total_eff_loss_mwh += yr_loss_mwh
        
        yearly_cycles = (yr_bess_mwh * 1000) / current_usable_kwh
        total_lifetime_cycles += yearly_cycles
        
        yearly_results.append({
            "Year": year, 
            "SoH (%)": round((current_usable_kwh/initial_kwh)*100, 1),
            "Cap (kWh)": round(current_usable_kwh, 1), 
            "Cycles": int(yearly_cycles),
            "Loss (MWh)": round(yr_loss_mwh, 2),
            "EV Load (MWh)": round(yr_ev_mwh, 1),
            "Util (%)": round((df['load_kw'].mean() / charger_cap) * 100, 1),
            "QoS Loss (%)": round(((missed_energy * year_mult / 1000) / yr_ev_mwh) * 100, 2),
            "Avg Plug/d": round((yr_bess_mwh * 1000) / (365 * num_plugs), 1)
        })
        df['grid_used'], df['bess_disc'] = grid_used, bess_disc
        plot_data[year] = df.copy()
        current_usable_kwh *= 0.985
        for _ in range(int(yearly_cycles)): 
            rate = 0.0045 if total_lifetime_cycles <= 800 else (0.0018 if total_lifetime_cycles <= 2500 else 0.0055)
            current_usable_kwh *= (1 - (rate / 100))

    # --- TOP KPIs ---
    st.markdown('<p class="kpi-header">KPIs after 10 years of operation</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cycles", int(total_lifetime_cycles))
    c2.metric("Final BESS SoH", f"{yearly_results[-1]['SoH (%)']}%")
    c3.metric("Total Loss", f"{total_eff_loss_mwh:.1f} MWh")
    c4.metric("Avg Plug Throughput", f"{np.mean([x['Avg Plug/d'] for x in yearly_results]):.1f} kWh/d")

    # --- CHART ---
    st.write("---")
    view_yr = st.selectbox("Year Profile:", list(range(1, 11)), label_visibility="collapsed")
    p_df = plot_data[view_yr]
    
    day_indices, unique_days = [], []
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        indices = p_df.index[p_df['day_name'] == day].tolist()
        if indices:
            day_indices.append(indices[len(indices)//2])
            unique_days.append(day)

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Grid', x=p_df.index, y=p_df['grid_used'], marker_color='#2b579a', opacity=0.8))
    fig.add_trace(go.Bar(name='BESS', x=p_df.index, y=p_df['bess_disc'], marker_color='#7eb26d'))
    fig.add_trace(go.Scatter(name='Demand', x=p_df.index, y=p_df['load_kw'], line=dict(color=KEMPOWER_ORANGE, width=1, dash='dot')))
    fig.update_layout(barmode='stack', margin=dict(l=0, r=0, t=5, b=20), height=410, yaxis_title="Power (kW)", xaxis=dict(tickmode='array', tickvals=day_indices, ticktext=unique_days), legend=dict(orientation="h", y=1.1, x=1))
    st.plotly_chart(fig, use_container_width=True)

    # --- BOTTOM SECTION ---
    st.write("---")
    res_col, qos_col = st.columns([2.2, 1])
    
    with res_col:
        st.markdown("**10-Year Roadmap**")
        final_df = pd.DataFrame(yearly_results).set_index("Year")
        st.dataframe(
            final_df,
            use_container_width=True,
            height=385,
            column_config={
                "Cycles": st.column_config.Column("Yr Cycles", width="small"),
                "Loss (MWh)": st.column_config.Column("Yr Loss", width="small"),
                "SoH (%)": st.column_config.Column(width="small"),
                "Plug Thr": st.column_config.Column("Plug kWh/d", width="small"),
            }
        )

    with qos_col:
        st.markdown("**Quality of Service (QoS) Legend**")
        st.markdown('<div class="qos-text"><b>0% - 1% Optimal:</b> High reliability. Monitoring only.</div>', unsafe_allow_html=True)
        st.markdown('<div class="qos-text"><b>2% - 5% Warning:</b> Risk of slow charging. Sizing check suggested.</div>', unsafe_allow_html=True)
        st.markdown('<div class="qos-text"><b>> 5% Critical:</b> Significant lost revenue. Augmentation required.</div>', unsafe_allow_html=True)
        
        # PDF GENERATOR
        def generate_pdf(results_df):
            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=A4)
            styles = getSampleStyleSheet()
            els = []
            if os.path.exists(LOGO_PATH):
                els.append(RLImage(LOGO_PATH, width=120, height=40))
                els.append(Spacer(1, 10))
            
            els.append(Paragraph("Kempower | BESS Sizing Report", styles['Title']))
            els.append(Spacer(1, 15))
            
            data = [["Year"] + results_df.columns.tolist()] + [[i] + row for i, row in zip(results_df.index, results_df.values.tolist())]
            t = Table(data)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor(KEMPOWER_ORANGE)),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('FONTSIZE', (0,0), (-1,-1), 8)
            ]))
            els.append(t)
            doc.build(els)
            return buf.getvalue()

        st.divider()
        st.download_button(
            label="📥 Export Strategic PDF Report",
            data=generate_pdf(final_df),
            file_name="Kempower_BESS_Sizing_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

else:
    st.info("Kempower | BESS Sizing Tool: Upload CSV to begin.")