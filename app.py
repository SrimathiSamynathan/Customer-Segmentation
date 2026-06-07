import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid #3d4466;
        margin: 5px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #4fc3f7;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #aab4c8;
        margin-top: 5px;
    }
    .segment-card {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px;
        padding: 15px;
        border-left: 4px solid #4fc3f7;
        margin: 8px 0;
    }
    h1, h2, h3 { color: #e0e6f0 !important; }
    .stSidebar { background-color: #161b2e; }
</style>
""", unsafe_allow_html=True)

# ── Segment Labels ────────────────────────────────────────────
SEGMENT_NAMES = {
    0: ("💎 Premium Loyalists", "#FFD700"),
    1: ("🛍️ Budget Shoppers", "#FF6B6B"),
    2: ("🎯 Target Customers", "#4FC3F7"),
    3: ("💤 Low Engagement", "#A0A0A0"),
    4: ("🚀 High Potential", "#7CFC00"),
}

# ── Header ────────────────────────────────────────────────────
st.markdown("## 👥 Customer Segmentation Dashboard")
st.markdown("*Segment customers based on behavior and demographics using K-Means Clustering*")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/crowd.png", width=80)
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])
    st.markdown("---")

    n_clusters = st.slider("🔢 Number of Clusters (K)", 2, 8, 5)
    st.markdown("---")

    st.markdown("### 🔍 Filters")
    gender_filter = st.multiselect("Gender", ["Male", "Female"], default=["Male", "Female"])
    age_range = st.slider("Age Range", 18, 70, (18, 70))
    income_range = st.slider("Annual Income (k$)", 15, 137, (15, 137))
    st.markdown("---")
    st.info("📌 Upload Mall_Customers.csv from Kaggle to get started!")

# ── Load Data ─────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

def preprocess(df):
    df = df.copy()
    if 'Genre' in df.columns:
        df.rename(columns={'Genre': 'Gender'}, inplace=True)
    df['Gender_Encoded'] = (df['Gender'] == 'Male').astype(int)
    return df

def run_kmeans(df, k):
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    df['Segment'] = df['Cluster'].map(lambda x: SEGMENT_NAMES.get(x, (f"Segment {x}", "#ffffff"))[0])
    return df, kmeans, scaler, X_scaled

# ── Main App ──────────────────────────────────────────────────
if uploaded_file:
    raw_df = load_data(uploaded_file)
    df = preprocess(raw_df)

    # Apply filters
    df = df[df['Gender'].isin(gender_filter)]
    df = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])]
    df = df[(df['Annual Income (k$)'] >= income_range[0]) & (df['Annual Income (k$)'] <= income_range[1])]

    df, kmeans, scaler, X_scaled = run_kmeans(df, n_clusters)

    # ── KPI Cards ─────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{len(df)}</div>
            <div class='metric-label'>👥 Total Customers</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{n_clusters}</div>
            <div class='metric-label'>🔢 Segments Found</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>${df['Annual Income (k$)'].mean():.0f}k</div>
            <div class='metric-label'>💰 Avg Income</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{df['Spending Score (1-100)'].mean():.0f}</div>
            <div class='metric-label'>🛍️ Avg Spending Score</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts Row 1 ──────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Income vs Spending (Clusters)")
        fig = px.scatter(
            df, x='Annual Income (k$)', y='Spending Score (1-100)',
            color='Segment', size='Age',
            hover_data=['Gender', 'Age'],
            color_discrete_sequence=px.colors.qualitative.Bold,
            template='plotly_dark'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🥧 Segment Distribution")
        seg_counts = df['Segment'].value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Count']
        fig2 = px.pie(
            seg_counts, names='Segment', values='Count',
            color_discrete_sequence=px.colors.qualitative.Bold,
            template='plotly_dark', hole=0.4
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Charts Row 2 ──────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### 📈 Elbow Method (Optimal K)")
        inertias = []
        K_range = range(1, 11)
        features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        X = StandardScaler().fit_transform(df[features])
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            inertias.append(km.inertia_)
        fig3 = px.line(
            x=list(K_range), y=inertias,
            markers=True, template='plotly_dark',
            labels={'x': 'Number of Clusters (K)', 'y': 'Inertia'}
        )
        fig3.add_vline(x=n_clusters, line_dash="dash", line_color="#4fc3f7")
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("### 👫 Gender Distribution per Segment")
        gender_seg = df.groupby(['Segment', 'Gender']).size().reset_index(name='Count')
        fig4 = px.bar(
            gender_seg, x='Segment', y='Count', color='Gender',
            barmode='group', template='plotly_dark',
            color_discrete_sequence=['#4FC3F7', '#FF6B6B']
        )
        fig4.update_layout(height=400, xaxis_tickangle=-20)
        st.plotly_chart(fig4, use_container_width=True)

    # ── 3D Scatter ────────────────────────────────────────────
    st.markdown("### 🌐 3D Cluster Visualization")
    fig5 = px.scatter_3d(
        df, x='Age', y='Annual Income (k$)', z='Spending Score (1-100)',
        color='Segment', symbol='Gender',
        color_discrete_sequence=px.colors.qualitative.Bold,
        template='plotly_dark', height=550
    )
    st.plotly_chart(fig5, use_container_width=True)

    # ── Segment Summary ───────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏷️ Segment Summary")
    summary = df.groupby('Segment').agg(
        Customers=('Cluster', 'count'),
        Avg_Age=('Age', 'mean'),
        Avg_Income=('Annual Income (k$)', 'mean'),
        Avg_Spending=('Spending Score (1-100)', 'mean')
    ).round(1).reset_index()

    for _, row in summary.iterrows():
        st.markdown(f"""<div class='segment-card'>
            <b>{row['Segment']}</b> &nbsp;|&nbsp;
            👥 {int(row['Customers'])} customers &nbsp;|&nbsp;
            🎂 Avg Age: {row['Avg_Age']} &nbsp;|&nbsp;
            💰 Avg Income: ${row['Avg_Income']}k &nbsp;|&nbsp;
            🛍️ Avg Spending: {row['Avg_Spending']}
        </div>""", unsafe_allow_html=True)

    # ── Predict New Customer ───────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔮 Predict Segment for New Customer")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        new_age = st.number_input("Age", 18, 100, 30)
    with col_b:
        new_income = st.number_input("Annual Income (k$)", 10, 200, 60)
    with col_c:
        new_spending = st.number_input("Spending Score (1-100)", 1, 100, 50)
    with col_d:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 Predict!", use_container_width=True)

    if predict_btn:
        new_data = scaler.transform([[new_age, new_income, new_spending]])
        pred_cluster = kmeans.predict(new_data)[0]
        seg_name, seg_color = SEGMENT_NAMES.get(pred_cluster, (f"Segment {pred_cluster}", "#ffffff"))
        st.success(f"This customer belongs to: **{seg_name}** (Cluster {pred_cluster})")

    # ── Download Button ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📥 Download Segmented Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download CSV with Segments",
        data=csv,
        file_name="segmented_customers.csv",
        mime="text/csv",
        use_container_width=True
    )

    # ── Raw Data ──────────────────────────────────────────────
    with st.expander("📋 View Raw Data"):
        st.dataframe(df, use_container_width=True)

else:
    # Welcome screen
    st.markdown("""
    <div style='text-align:center; padding: 60px 20px;'>
        <h1 style='font-size:4rem;'>👥</h1>
        <h2>Welcome to Customer Segmentation Dashboard</h2>
        <p style='color:#aab4c8; font-size:1.1rem;'>
            Upload your <b>Mall_Customers.csv</b> file from the sidebar to get started!
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📊 **Interactive Charts**\nScatter, Pie, Bar & 3D visualizations")
    with col2:
        st.info("🔮 **Predict Segments**\nFind which segment a new customer belongs to")
    with col3:
        st.info("📥 **Download Results**\nExport segmented data as CSV")