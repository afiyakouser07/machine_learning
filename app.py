import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="User Analytics Dashboard",
    layout="wide"
)

df = pd.read_csv("data/user_behavior_dataset.csv")

st.title("📱 User Behavior Analytics Dashboard")

# KPIs
col1,col2,col3,col4 = st.columns(4)

col1.metric("Users", len(df))
col2.metric("Avg Screen Time",
            round(df["Screen On Time (hours/day)"].mean(),2))
col3.metric("Avg Battery Drain",
            round(df["Battery Drain (mAh/day)"].mean(),2))
col4.metric("Avg Data Usage",
            round(df["Data Usage (MB/day)"].mean(),2))

st.divider()

# Age Distribution
fig1 = px.histogram(
    df,
    x="Age",
    nbins=20,
    title="Age Distribution"
)

st.plotly_chart(fig1,use_container_width=True)

# Gender Distribution
fig2 = px.pie(
    df,
    names="Gender",
    title="Gender Distribution"
)

st.plotly_chart(fig2,use_container_width=True)

# Device Model
fig3 = px.bar(
    df["Device Model"].value_counts(),
    title="Device Popularity"
)

st.plotly_chart(fig3,use_container_width=True)

# Correlation Heatmap
corr = df.select_dtypes(include='number').corr()

fig4 = px.imshow(
    corr,
    text_auto=True,
    title="Correlation Matrix"
)

st.plotly_chart(fig4,use_container_width=True)
