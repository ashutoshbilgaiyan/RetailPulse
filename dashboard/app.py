
import streamlit as st
import pandas as pd
import warnings
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="RetailPulse",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── paths ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed"
MODELS_PATH = BASE_DIR / "models"

# ── load data ──────────────────────────────────────
@st.cache_data
def load_data():
    rfm = pd.read_csv(DATA_PATH / "rfm_segments.csv")
    churn = pd.read_csv(DATA_PATH / "churn_scores.csv")
    forecast = pd.read_csv(DATA_PATH / "demand_forecast.csv")
    forecast["ds"] = pd.to_datetime(forecast["ds"])
    inventory = pd.read_csv(DATA_PATH / "inventory_recommendations.csv")
    inventory["date"] = pd.to_datetime(inventory["date"])
    segments = pd.read_csv(DATA_PATH / "segment_report.csv")
    return rfm, churn, forecast, inventory, segments

@st.cache_resource
def load_models():
    model = joblib.load(MODELS_PATH / "xgboost_churn.pkl")
    explainer = joblib.load(MODELS_PATH / "shap_explainer.pkl")
    return model, explainer

rfm, churn, forecast, inventory, segments = load_data()
model, explainer = load_models()

# ── sidebar ────────────────────────────────────────
st.sidebar.markdown("# RetailPulse")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "📈 Demand Forecasting",
        "⚠️ Churn Prediction",
        "👥 Customer Segments",
        "📦 Inventory"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**RetailPulse v1.0**")
st.sidebar.caption("Built by Ashutosh Bilgaiyan")
st.sidebar.markdown("---")

# optional export
st.sidebar.subheader("📥 Export Reports")

full_report = pd.merge(
    churn[["Customer ID", "segment", "churn_risk", "recency", "frequency", "monetary"]],
    inventory[["date", "forecast", "reorder_qty", "alert"]].rename(columns={"date": "report_date"}),
    how="cross"
)

csv = full_report.head(1000).to_csv(index=False)

st.sidebar.download_button(
    label="📥 Download Summary Report",
    data=csv,
    file_name="retailpulse_report.csv",
    mime="text/csv"
)

# ── OVERVIEW ───────────────────────────────────────
if page == "🏠 Overview":
    st.title("📊 RetailPulse — AI Analytics Dashboard")
    st.markdown("*End-to-end retail analytics powered by ML*")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(rfm):,}")
    col2.metric(
        "High Churn Risk",
        f"{(churn['churn_risk'] > 0.7).sum():,}",
        delta="-needs attention",
        delta_color="inverse"
    )
    col3.metric("Champion Customers", f"{(rfm['segment'] == 'Champions').sum():,}")
    col4.metric("Avg Churn Risk", f"{churn['churn_risk'].mean():.1%}")

    st.markdown("---")
    st.subheader("Quick Insights")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            segments,
            values="customer_count",
            names="segment",
            title="Customer Segments",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            segments.sort_values("revenue_share", ascending=False),
            x="segment",
            y="revenue_share",
            title="Revenue Share by Segment (%)",
            color="segment"
        )
        st.plotly_chart(fig, use_container_width=True)

# ── DEMAND FORECASTING ─────────────────────────────
elif page == "📈 Demand Forecasting":
    st.title("📈 Demand Forecasting")
    st.markdown("30-day demand and revenue forecast using Prophet")
    st.markdown("---")

    st.subheader("What-If Analysis")
    growth_factor = st.slider(
        "Adjust forecast by (%)",
        min_value=-30,
        max_value=50,
        value=0,
        step=5
    )

    future = forecast[forecast["ds"] > pd.Timestamp("2011-11-09")].copy()
    future["yhat_adjusted"] = future["yhat"] * (1 + growth_factor / 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"],
        name="Forecast"
    ))
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_upper"],
        name="Upper bound",
        line=dict(dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_lower"],
        name="Lower bound",
        line=dict(dash="dash"),
        fill="tonexty"
    ))

    if growth_factor != 0:
        fig.add_trace(go.Scatter(
            x=future["ds"],
            y=future["yhat_adjusted"],
            name=f"Adjusted ({growth_factor:+}%)",
            line=dict(dash="dot")
        ))

    fig.update_layout(
        title="30-Day Demand Forecast",
        xaxis_title="Date",
        yaxis_title="Revenue (£)",
        hovermode="x unified",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Daily Forecast", f"£{future['yhat'].mean():,.0f}")
    col2.metric("Peak Day Forecast", f"£{future['yhat'].max():,.0f}")
    col3.metric("Total 30-Day Revenue", f"£{future['yhat'].sum():,.0f}")

    st.subheader("Daily Forecast Table")
    display_forecast = future[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
        columns={
            "ds": "Date",
            "yhat": "Forecast",
            "yhat_lower": "Lower",
            "yhat_upper": "Upper"
        }
    )
    st.dataframe(display_forecast, use_container_width=True)

    csv = future[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(index=False)
    st.download_button("📥 Download Forecast CSV", csv, "forecast.csv", "text/csv")

# ── CHURN PREDICTION ───────────────────────────────
elif page == "⚠️ Churn Prediction":
    st.title("⚠️ Churn Prediction")
    st.caption("Customer churn-risk analysis powered by XGBoost")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        risk_threshold = st.slider("Minimum churn risk (%)", 0, 100, 70, 5) / 100
    with col2:
        segment_filter = st.multiselect(
            "Filter by segment",
            options=churn["segment"].unique().tolist(),
            default=churn["segment"].unique().tolist()
        )

    filtered = churn[
        (churn["churn_risk"] >= risk_threshold) &
        (churn["segment"].isin(segment_filter))
    ].copy()

    col1, col2, col3 = st.columns(3)
    col1.metric("Customers at Risk", f"{len(filtered):,}")
    col2.metric(
        "Avg Risk Score",
        f"{filtered['churn_risk'].mean():.1%}" if len(filtered) > 0 else "N/A"
    )
    col3.metric(
        "Highest Risk",
        f"{filtered['churn_risk'].max():.1%}" if len(filtered) > 0 else "N/A"
    )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            churn,
            x="churn_risk",
            nbins=30,
            title="Churn Risk Distribution"
        )
        fig.add_vline(
            x=risk_threshold,
            line_dash="dash",
            annotation_text=f"Threshold: {risk_threshold:.0%}"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        seg_risk = churn.groupby("segment")["churn_risk"].mean().reset_index()
        fig = px.bar(
            seg_risk.sort_values("churn_risk", ascending=False),
            x="segment",
            y="churn_risk",
            title="Avg Churn Risk by Segment",
            color="churn_risk"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"At-Risk Customers ({len(filtered):,})")
    if len(filtered) > 0:
        display_df = filtered[
            ["Customer ID", "segment", "recency", "frequency", "monetary", "churn_risk"]
        ].sort_values("churn_risk", ascending=False)
        st.dataframe(display_df, use_container_width=True)

        csv = filtered.to_csv(index=False)
        st.download_button(
            "📥 Download At-Risk Customers",
            csv,
            "churn_risk.csv",
            "text/csv"
        )
    else:
        st.info("No customers match the selected filters")

# ── CUSTOMER SEGMENTS ──────────────────────────────
elif page == "👥 Customer Segments":
    st.title("👥 Customer Segmentation")
    st.caption("Customer segmentation using RFM analysis and K-Means clustering")
    st.markdown("---")

    selected_seg = st.selectbox(
        "Select segment to analyse",
        ["All"] + rfm["segment"].unique().tolist()
    )

    display_rfm = rfm if selected_seg == "All" else rfm[rfm["segment"] == selected_seg]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers", f"{len(display_rfm):,}")
    col2.metric("Avg Recency", f"{display_rfm['recency'].mean():.0f} days")
    col3.metric("Avg Orders", f"{display_rfm['frequency'].mean():.1f}")
    col4.metric("Avg Spend", f"£{display_rfm['monetary'].mean():,.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            display_rfm,
            x="recency",
            y="frequency",
            color="segment",
            size="monetary",
            hover_data=["Customer ID", "monetary"],
            title="Recency vs Frequency (size = spend)",
            opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            segments.sort_values("revenue_share", ascending=True),
            x="revenue_share",
            y="segment",
            orientation="h",
            title="Revenue Share by Segment (%)",
            text="revenue_share"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment Business Report")
    st.dataframe(segments, use_container_width=True)

# ── INVENTORY ──────────────────────────────────────
elif page == "📦 Inventory":
    st.title("📦 Inventory Optimization")
    st.caption("Inventory recommendations and low-stock alerts based on forecasted demand")
    st.markdown("---")

    low_stock = inventory[inventory["alert"] == "🔴 LOW STOCK"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Forecast Days", f"{len(inventory)}")
    col2.metric(
        "Low Stock Alerts",
        f"{len(low_stock)}",
        delta=f"{len(low_stock)} days need attention",
        delta_color="inverse"
    )
    col3.metric("Avg Daily Reorder", f"{inventory['reorder_qty'].mean():,.0f} units")

    st.markdown("---")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=inventory["date"],
        y=inventory["reorder_qty"],
        name="Reorder Quantity"
    ))
    fig.add_trace(go.Scatter(
        x=inventory["date"],
        y=inventory["forecast"],
        name="Forecast"
    ))
    fig.update_layout(
        title="Daily Reorder Quantities vs Forecast",
        xaxis_title="Date",
        yaxis_title="Units / Revenue (£)",
        hovermode="x unified",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    if len(low_stock) > 0:
        st.subheader("🔴 Low Stock Alerts")
        st.dataframe(
            low_stock[["date", "forecast", "reorder_qty", "alert"]],
            use_container_width=True
        )

    st.subheader("Full Inventory Recommendations")
    st.dataframe(inventory.sort_values("date"), use_container_width=True)

    csv = inventory.to_csv(index=False)
    st.download_button(
        "📥 Download Inventory Plan",
        csv,
        "inventory_plan.csv",
        "text/csv"
    )

