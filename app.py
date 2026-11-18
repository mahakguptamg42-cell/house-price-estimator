import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Attempt to load Plotly; fall back to Altair/Streamlit charts if Plotly is not installed
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    try:
        import altair as alt
        HAS_ALTAIR = True
    except ImportError:
        HAS_ALTAIR = False

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Indian Real Estate Price Estimator",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Ultra-Clean Professional UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white !important;
        padding: 28px;
        border-radius: 18px;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.4);
        margin-bottom: 20px;
    }
    .metric-label {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #e2e8f0 !important;
        font-weight: 600;
    }
    .metric-value-cr {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 8px 0;
        color: #ffffff !important;
    }
    .metric-value-lakh {
        font-size: 1.35rem;
        font-weight: 600;
        color: #f1f5f9 !important;
    }
    .info-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.9rem;
        margin-top: 10px;
        color: #ffffff !important;
        font-weight: 500;
    }
    .stat-box {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .stat-title {
        color: #475569 !important;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .stat-val {
        color: #0f172a !important;
        font-size: 1.4rem;
        font-weight: 800;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_trained_model():
    model_paths = ['best_indian_house_model.pkl', 'model.pkl']
    for p in model_paths:
        if os.path.exists(p):
            try:
                return joblib.load(p)
            except Exception:
                continue
    return None

model = load_trained_model()

# ==================== SIDEBAR INPUTS ====================
st.sidebar.markdown("## ⚙️ Property Configuration")
st.sidebar.markdown("Adjust property features to calculate real-time valuation:")

location_tier_display = st.sidebar.selectbox(
    "📍 Location / City Tier",
    options=[
        "Metro (Mumbai, Delhi NCR, Bengaluru)",
        "Tier-1 (Pune, Hyderabad, Jaipur)",
        "Tier-2 (Indore, Lucknow, Chandigarh)"
    ],
    index=0
)

tier_mapping = {
    "Metro (Mumbai, Delhi NCR, Bengaluru)": "Metro (Mumbai/Delhi/Bengaluru)",
    "Tier-1 (Pune, Hyderabad, Jaipur)": "Tier-1 (Pune/Hyderabad/Jaipur)",
    "Tier-2 (Indore, Lucknow, Chandigarh)": "Tier-2 (Indore/Lucknow/Chandigarh)"
}
location_tier = tier_mapping[location_tier_display]

carpet_area = st.sidebar.slider(
    "📐 Carpet Area (sq. ft.)",
    min_value=400,
    max_value=4500,
    value=1250,
    step=25
)

bhk = st.sidebar.slider(
    "🛏️ Bedrooms (BHK)",
    min_value=1,
    max_value=5,
    value=3,
    step=1
)

bathrooms = st.sidebar.slider(
    "🚿 Bathrooms",
    min_value=1,
    max_value=5,
    value=2,
    step=1
)

property_age = st.sidebar.slider(
    "🏗️ Property Age (years)",
    min_value=0,
    max_value=30,
    value=3,
    step=1
)

furnishing_status = st.sidebar.selectbox(
    "🛋️ Furnishing Status",
    options=["Unfurnished", "Semi-Furnished", "Fully-Furnished"],
    index=1
)

parking_choice = st.sidebar.radio(
    "🚗 Reserved Parking",
    options=["Yes", "No"],
    index=0,
    horizontal=True
)

parking_slots = 1 if parking_choice == "Yes" else 0

# ==================== MAIN PANEL ====================
st.markdown('<div class="main-title">🏡 Real-Time Indian House Price Estimator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced Machine Learning Valuation Dashboard for Residential Properties</div>', unsafe_allow_html=True)

input_data = pd.DataFrame([{
    'location_tier': location_tier,
    'carpet_area_sqft': carpet_area,
    'bhk': bhk,
    'bathrooms': bathrooms,
    'property_age': property_age,
    'furnishing_status': furnishing_status,
    'parking': parking_slots
}])

try:
    if model is not None:
        pred_lakh = float(model.predict(input_data)[0])
        pred_lakh = max(15.0, pred_lakh)
    else:
        base_rate = 14000 if "Metro" in location_tier else (9000 if "Tier-1" in location_tier else 6000)
        pred_lakh = (carpet_area * base_rate) / 100000.0
except Exception:
    pred_lakh = 95.0

pred_crores = pred_lakh / 100.0
price_per_sqft = (pred_lakh * 100000.0) / carpet_area

loan_amount = (pred_lakh * 100000.0) * 0.80
r = 0.085 / 12.0
n = 20 * 12
emi = loan_amount * r * ((1 + r)**n) / (((1 + r)**n) - 1)

# ----------------- Hero Metric Card -----------------
st.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Estimated Market Valuation</div>
    <div class="metric-value-cr">
        {"₹ " + f"{pred_crores:.2f} Crores" if pred_lakh >= 100.0 else "₹ " + f"{pred_lakh:.2f} Lakhs"}
    </div>
    <div class="metric-value-lakh">
        Equivalent: ₹ {pred_lakh:.2f} Lakhs &nbsp;•&nbsp; (~₹ {price_per_sqft:,.0f} / sq. ft.)
    </div>
    <div class="info-badge">
        🎯 {bhk} BHK &nbsp;|&nbsp; {carpet_area:,} sq. ft. &nbsp;|&nbsp; {location_tier_display.split(' ')[0]} &nbsp;|&nbsp; {furnishing_status}
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- Quick Stat Boxes -----------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-title">Base Rate / sq. ft.</div>
        <div class="stat-val">₹{price_per_sqft:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-title">Est. Monthly EMI</div>
        <div class="stat-val">₹{emi:,.0f}/mo</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-title">Property Age</div>
        <div class="stat-val">{"Brand New" if property_age == 0 else f"{property_age} Years"}</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-title">Parking Spaces</div>
        <div class="stat-val">{parking_slots} Reserved</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.markdown("### 📊 Market Insights & Interactive Analytics")

tab1, tab2 = st.tabs(["📈 Price vs. Carpet Area across City Tiers", "🏙️ Cross-City Price Comparison"])

with tab1:
    area_range = np.linspace(400, 4500, 35).astype(int)
    tiers = [
        "Metro (Mumbai/Delhi/Bengaluru)",
        "Tier-1 (Pune/Hyderabad/Jaipur)",
        "Tier-2 (Indore/Lucknow/Chandigarh)"
    ]
    tier_labels = {
        "Metro (Mumbai/Delhi/Bengaluru)": "Metro (Mumbai / Delhi NCR / Bengaluru)",
        "Tier-1 (Pune/Hyderabad/Jaipur)": "Tier-1 (Pune / Hyderabad / Jaipur)",
        "Tier-2 (Indore/Lucknow/Chandigarh)": "Tier-2 (Indore / Lucknow / Chandigarh)"
    }
    
    chart_data_rows = []
    for t in tiers:
        for a in area_range:
            chart_data_rows.append({
                'location_tier': t,
                'carpet_area_sqft': a,
                'bhk': bhk,
                'bathrooms': bathrooms,
                'property_age': property_age,
                'furnishing_status': furnishing_status,
                'parking': parking_slots
            })
            
    curve_df = pd.DataFrame(chart_data_rows)
    if model is not None:
        try:
            curve_df['price_lakh'] = model.predict(curve_df)
        except Exception:
            curve_df['price_lakh'] = curve_df['carpet_area_sqft'] * 0.08
    else:
        curve_df['price_lakh'] = curve_df['carpet_area_sqft'] * 0.08
        
    curve_df['Tier Label'] = curve_df['location_tier'].map(tier_labels)
    
    if HAS_PLOTLY:
        fig = px.line(
            curve_df,
            x='carpet_area_sqft',
            y='price_lakh',
            color='Tier Label',
            title=f"Price Scaling with Carpet Area ({bhk} BHK, {furnishing_status})",
            labels={'carpet_area_sqft': 'Carpet Area (sq. ft.)', 'price_lakh': 'Estimated Price (₹ Lakhs)'},
            color_discrete_map={
                "Metro (Mumbai / Delhi NCR / Bengaluru)": "#1e3a8a",
                "Tier-1 (Pune / Hyderabad / Jaipur)": "#0284c7",
                "Tier-2 (Indore / Lucknow / Chandigarh)": "#059669"
            }
        )
        fig.add_trace(go.Scatter(
            x=[carpet_area],
            y=[pred_lakh],
            mode='markers+text',
            name='Current Selection',
            text=[f"Current: ₹{pred_lakh:.1f}L"],
            textposition="top center",
            marker=dict(size=14, color='#dc2626', symbol='diamond', line=dict(width=2, color='white'))
        ))
        fig.update_layout(
            hovermode="x unified",
            legend=dict(
                orientation="h", 
                yanchor="top", 
                y=-0.25, 
                xanchor="center", 
                x=0.5,
                title=dict(text="")
            ),
            margin=dict(l=20, r=20, t=70, b=90),
            title=dict(font=dict(size=16), y=0.95, x=0.0)
        )
        st.plotly_chart(fig, use_container_width=True)
    elif HAS_ALTAIR:
        chart = alt.Chart(curve_df).mark_line(size=3).encode(
            x=alt.X('carpet_area_sqft:Q', title='Carpet Area (sq. ft.)'),
            y=alt.Y('price_lakh:Q', title='Estimated Price (₹ Lakhs)'),
            color=alt.Color('Tier Label:N', legend=alt.Legend(title="City Tier")),
            tooltip=['Tier Label', 'carpet_area_sqft', alt.Tooltip('price_lakh:Q', format='.2f')]
        ).properties(height=420)
        st.altair_chart(chart, use_container_width=True)

with tab2:
    tier_comp_rows = []
    for t in tiers:
        tier_comp_rows.append({
            'location_tier': t,
            'carpet_area_sqft': carpet_area,
            'bhk': bhk,
            'bathrooms': bathrooms,
            'property_age': property_age,
            'furnishing_status': furnishing_status,
            'parking': parking_slots
        })
    tier_comp_df = pd.DataFrame(tier_comp_rows)
    if model is not None:
        try:
            tier_comp_df['price_lakh'] = model.predict(tier_comp_df)
        except Exception:
            tier_comp_df['price_lakh'] = [95.0, 70.0, 50.0]
    else:
        tier_comp_df['price_lakh'] = [95.0, 70.0, 50.0]
        
    tier_comp_df['City Tier'] = tier_comp_df['location_tier'].map(tier_labels)
    tier_comp_df['Price (₹ Cr)'] = (tier_comp_df['price_lakh'] / 100.0).round(2)
    tier_comp_df['Rate (₹/sqft)'] = ((tier_comp_df['price_lakh'] * 100000) / carpet_area).round(0)
    
    if HAS_PLOTLY:
        fig_bar = px.bar(
            tier_comp_df,
            x='City Tier',
            y='price_lakh',
            color='City Tier',
            text=tier_comp_df['price_lakh'].apply(lambda x: f"₹{x:.1f} L" if x < 100 else f"₹{x/100:.2f} Cr"),
            labels={'price_lakh': 'Estimated Valuation (₹ Lakhs)'},
            color_discrete_sequence=['#1e3a8a', '#0284c7', '#059669']
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis=dict(title='Valuation (₹ Lakhs)')
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    elif HAS_ALTAIR:
        bar_chart = alt.Chart(tier_comp_df).mark_bar().encode(
            x=alt.X('City Tier:N', title='City Tier'),
            y=alt.Y('price_lakh:Q', title='Valuation (₹ Lakhs)'),
            color=alt.Color('City Tier:N', legend=None),
            tooltip=['City Tier', alt.Tooltip('price_lakh:Q', format='.2f')]
        ).properties(height=380)
        st.altair_chart(bar_chart, use_container_width=True)
        
    st.dataframe(
        tier_comp_df[['City Tier', 'price_lakh', 'Price (₹ Cr)', 'Rate (₹/sqft)']].rename(columns={'price_lakh': 'Price (₹ Lakhs)'}),
        use_container_width=True
    )
