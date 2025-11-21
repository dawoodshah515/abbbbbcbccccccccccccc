# app.py
import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
from fpdf import FPDF
from dotenv import load_dotenv

# ---------------- Load API (optional) ----------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Multi-Agent Dataset Dashboard", layout="wide")
st.title("🤖 Multi-Agent Dataset Dashboard & Insights")

# ================= Upload Dataset =================
uploaded_file = st.file_uploader("Upload CSV dataset", type="csv")
if not uploaded_file:
    st.info("Please upload a CSV dataset.")
    st.stop()

# ================= Data Cleaning Agent =================
@st.cache_data
def clean_data(df):
    df_cleaned = df.copy()
    for col in df_cleaned.columns:
        if pd.api.types.is_numeric_dtype(df_cleaned[col]):
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
            df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
        else:
            df_cleaned[col] = df_cleaned[col].astype(str)
            df_cleaned[col].fillna(df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else "Unknown", inplace=True)
    return df_cleaned

df = pd.read_csv(uploaded_file)
df_cleaned = clean_data(df)
dataset_name = uploaded_file.name.replace(" ", "_")
st.success(f"✅ Dataset `{uploaded_file.name}` uploaded and cleaned!")

# ================= Sidebar Filters (Top 3-4 only) =================
st.sidebar.header("Important Filters")
numeric_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df_cleaned.select_dtypes(include=['object','category']).columns.tolist()

# Select top 2 numeric columns by variance
num_variance = df_cleaned[numeric_cols].var()
top_numeric = num_variance.sort_values(ascending=False).head(2).index.tolist()

# Select top 2 categorical columns by unique values (≤10)
cat_unique = {col: df_cleaned[col].nunique() for col in categorical_cols if df_cleaned[col].nunique() <= 10}
top_categorical = list(cat_unique.keys())[:2]

# Sidebar filters
filters = {}
for col in top_categorical:
    options = df_cleaned[col].unique().tolist()
    selected = st.sidebar.multiselect(f"Select {col}", options, default=options)
    filters[col] = selected

for col in top_numeric:
    min_val = float(df_cleaned[col].min())
    max_val = float(df_cleaned[col].max())
    selected = st.sidebar.slider(f"{col} range", min_val, max_val, (min_val, max_val))
    filters[col] = selected

# Apply filters
df_filtered = df_cleaned.copy()
for col, val in filters.items():
    if col in top_categorical:
        df_filtered = df_filtered[df_filtered[col].isin(val)]
    elif col in top_numeric:
        df_filtered = df_filtered[(df_filtered[col] >= val[0]) & (df_filtered[col] <= val[1])]

st.subheader("📊 Filtered Dataset Preview (first 4 rows)")
st.dataframe(df_filtered.head(4))

# ================= Insights Agent =================
st.subheader("📈 Dataset Insights")

# Numeric KPIs
st.markdown("### Numeric KPIs")
for col in top_numeric:
    mean_val = df_filtered[col].mean()
    sum_val = df_filtered[col].sum()
    max_val = df_filtered[col].max()
    min_val = df_filtered[col].min()
    st.metric(f"{col} Mean", f"{mean_val:.2f}")
    st.metric(f"{col} Sum", f"{sum_val:.2f}")
    st.metric(f"{col} Max", f"{max_val:.2f}")
    st.metric(f"{col} Min", f"{min_val:.2f}")

# Categorical top values
st.markdown("### Top Categorical Values")
for col in top_categorical:
    top = df_filtered[col].value_counts().head(4)
    st.write(f"**{col}:**")
    st.table(top)

# ================= Visualization Agent =================
st.subheader("📊 Visualizations")

# Pie chart for categorical
for col in top_categorical:
    fig = px.pie(df_filtered, names=col, values=df_filtered[col].map(lambda x:1),
                 title=f"{col} Distribution", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# Histogram & boxplots for numeric
for col in top_numeric:
    fig_hist = px.histogram(df_filtered, x=col, nbins=20, title=f"{col} Histogram", template="plotly_dark")
    st.plotly_chart(fig_hist, use_container_width=True)
    fig_box = px.box(df_filtered, y=col, title=f"{col} Boxplot", template="plotly_dark")
    st.plotly_chart(fig_box, use_container_width=True)

# Scatter plot between top 2 numeric
if len(top_numeric) == 2:
    fig = px.scatter(df_filtered, x=top_numeric[0], y=top_numeric[1],
                     color=top_categorical[0] if top_categorical else None,
                     title=f"{top_numeric[0]} vs {top_numeric[1]}", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# Correlation heatmap
if top_numeric:
    corr = df_filtered[top_numeric].corr()
    fig = ff.create_annotated_heatmap(z=corr.values, x=list(corr.columns), y=list(corr.columns),
                                      colorscale='Viridis')
    st.plotly_chart(fig, use_container_width=True)

# ================= PDF Agent =================
st.subheader("📄 Download Insights as PDF")

def create_pdf(df_preview, numeric, categorical):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Dataset Preview - {uploaded_file.name}", ln=True)
    for i, row in df_preview.iterrows():
        pdf.cell(0, 8, str(row.to_dict()), ln=True)
    pdf.cell(0, 10, "Numeric KPIs:", ln=True)
    for col in numeric:
        pdf.cell(0, 8, f"{col} Mean: {df_preview[col].mean():.2f}, Sum: {df_preview[col].sum():.2f}", ln=True)
    pdf.cell(0, 10, "Top Categorical Values:", ln=True)
    for col in categorical:
        top = df_preview[col].value_counts().head(4)
        pdf.cell(0, 8, f"{col}: {top.to_dict()}", ln=True)
    pdf_bytes = io.BytesIO()
    pdf.output(pdf_bytes)
    pdf_bytes.seek(0)
    return pdf_bytes

if st.button("📥 Download Insights PDF"):
    pdf_file = create_pdf(df_filtered.head(4), top_numeric, top_categorical)
    st.download_button("Download PDF", pdf_file, file_name=f"{dataset_name}_insights.pdf", mime="application/pdf")

# ================= Dashboard.py Agent =================
st.subheader("💾 Download Dashboard.py")

def create_dashboard_code(df_head, numeric, categorical):
    code = f"""
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Auto Dashboard - {uploaded_file.name}")
df = pd.DataFrame({df_head.to_dict()})

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Numeric KPIs
st.subheader("Numeric KPIs")
"""
    for col in numeric:
        code += f"""
st.metric("{col} Mean", df['{col}'].mean())
st.metric("{col} Sum", df['{col}'].sum())
st.metric("{col} Max", df['{col}'].max())
st.metric("{col} Min", df['{col}'].min())
"""
    for col in categorical:
        code += f"""
st.subheader("Top {col} Values")
st.table(df['{col}'].value_counts().head(4))
"""
    # Plots
    for col in numeric:
        code += f"""
fig = px.histogram(df, x='{col}', nbins=20, title='{col} Histogram')
st.plotly_chart(fig, use_container_width=True)
fig_box = px.box(df, y='{col}', title='{col} Boxplot')
st.plotly_chart(fig_box, use_container_width=True)
"""
    if len(numeric) == 2:
        code += f"""
fig = px.scatter(df, x='{numeric[0]}', y='{numeric[1]}', 
                 color='{categorical[0]}' if '{categorical}' else None,
                 title='{numeric[0]} vs {numeric[1]}')
st.plotly_chart(fig, use_container_width=True)
"""
    return code

if st.button("📥 Download Dashboard.py"):
    dashboard_code = create_dashboard_code(df_filtered.head(4), top_numeric, top_categorical)
    st.download_button("Download Dashboard.py", dashboard_code.encode("utf-8"),
                       file_name=f"{dataset_name}_dashboard.py", mime="text/x-python")
