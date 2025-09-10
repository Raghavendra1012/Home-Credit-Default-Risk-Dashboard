import streamlit as st
import matplotlib.pyplot as plt
from utils.load_data import load_and_preprocess
from utils.filters import global_filters
import numpy as np
import pandas as pd

# Load and preprocess data
df = load_and_preprocess()

# Apply global filters
filtered_df = global_filters(df)

# Show filtered data
st.write("Filtered Data", filtered_df.head(10))

st.title("📊 Overview & Data Quality")

#============= KPIs ==============
total_applicants = filtered_df['SK_ID_CURR'].nunique()
default_rate = filtered_df['TARGET'].mean() * 100
repaid_rate = (1 - filtered_df['TARGET'].mean()) * 100
total_features = filtered_df.shape[1]
avg_missing_per_feature = filtered_df.isnull().mean().mean() * 100
numerical_features = filtered_df.select_dtypes(include=['int', 'float']).columns
categorical_features = filtered_df.select_dtypes(include=['object']).columns
median_age = filtered_df['AGE_YEARS'].median()
median_annual_income = filtered_df['AMT_INCOME_TOTAL'].median()
avg_credit_amount = filtered_df['AMT_CREDIT'].mean()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Applicants", total_applicants)
col2.metric("Default Rate", f"{default_rate:.2f}%")
col3.metric("Repaid Rate", f"{repaid_rate:.2f}%")
col4.metric("Total Features", total_features)
col5.metric("Avg Missing %", f"{avg_missing_per_feature:.2f}%")

col6, col7, col8, col9, col10 = st.columns(5)
col6.metric("Numerical Features", f"{len(numerical_features):,}")
col7.metric("Categorical Features", f"{len(categorical_features):,}")
col8.metric("Median Age", f"{median_age:,.2f} years")
col9.metric("Median Annual Income", f"${median_annual_income:,.2f}")
col10.metric("Avg Credit Amount", f"${avg_credit_amount:,.2f}")

st.markdown("---")

# Dark theme for plots
plt.style.use('dark_background')

# Colors used
colors = {
    'target': ["#1f77b4", "#ff7f0e"],
    'missing': "#2ca02c",
    'age': "#b78824",
    'income': "#17becf",
    'credit': "#d77575",
    'box_income': "#17becf",
    'box_credit': "#d77575",
    'gender': ["#cb803f", "#5aa1d4", "#17becf"],
    'family': "#7fcb7f",
    'education': "#7846a5"
}

# Utility function to disable vertical grid lines and spines
def style_axis(ax):
    ax.grid(True, which='both', axis='y', linestyle='-', color='gray')
    ax.grid(False, axis='x')  # Disable vertical grid lines
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

# Plotting starts here
col1, col2 = st.columns(2)

# Pie / Donut — Target distribution
with col1:
    target_counts = filtered_df['TARGET'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(5, 4.5))
    ax1.pie(
        target_counts,
        labels=['Repaid (0)', 'Default (1)'],
        autopct='%1.1f%%',
        startangle=90,
        colors=colors['target'],
        textprops={'color': 'white'}
    )
    ax1.set_title("Target Distribution", fontsize=14, color='white')
    st.pyplot(fig1)

# Bar — Top 20 missing features
with col2:
    missing_features = filtered_df.isnull().mean().sort_values(ascending=False).head(20) * 100
    if missing_features.sum() == 0:
        st.write("✅ No missing features!")
    else:
        fig2, ax2 = plt.subplots(figsize=(5, 4.5))
        ax2.barh(missing_features.index[::-1], missing_features.values[::-1], color=colors['missing'])
        ax2.set_title("Top 20 Missing Features", fontsize=14, color='white')
        ax2.set_xlabel("Missing %", fontsize=12, color='white')
        ax2.tick_params(axis='x', colors='white')
        ax2.tick_params(axis='y', colors='white')
        style_axis(ax2)
        st.pyplot(fig2)

col3, col4 = st.columns(2)

# Histogram — Age
with col3:
    fig3, ax3 = plt.subplots(figsize=(5, 4.5))
    ax3.hist(filtered_df['AGE_YEARS'].dropna(), bins=30, color=colors['age'], edgecolor='white')
    ax3.set_title("Age Distribution", fontsize=14, color='white')
    ax3.set_xlabel("Age", fontsize=12, color='white')
    ax3.set_ylabel("Frequency", fontsize=12, color='white')
    ax3.tick_params(axis='x', colors='white')
    ax3.tick_params(axis='y', colors='white')
    style_axis(ax3)
    st.pyplot(fig3)

# Histogram — Annual Income
with col4:
    fig4, ax4 = plt.subplots(figsize=(5, 4.5))
    ax4.hist(filtered_df['AMT_INCOME_TOTAL'].dropna(), bins=30, color=colors['income'], edgecolor='white')
    ax4.set_title("Annual Income Distribution", fontsize=14, color='white')
    ax4.set_xlabel("Income", fontsize=12, color='white')
    ax4.set_ylabel("Frequency", fontsize=12, color='white')
    ax4.tick_params(axis='x', colors='white')
    ax4.tick_params(axis='y', colors='white')
    style_axis(ax4)
    st.pyplot(fig4)

col5, col6 = st.columns(2)

# Histogram — Credit Amount
with col5:
    fig5, ax5 = plt.subplots(figsize=(5, 4.5))
    ax5.hist(filtered_df['AMT_CREDIT'].dropna(), bins=30, color=colors['credit'], edgecolor='white')
    ax5.set_title("Credit Amount Distribution", fontsize=14, color='white')
    ax5.set_xlabel("Credit Amount", fontsize=12, color='white')
    ax5.set_ylabel("Frequency", fontsize=12, color='white')
    ax5.tick_params(axis='x', colors='white')
    ax5.tick_params(axis='y', colors='white')
    style_axis(ax5)
    st.pyplot(fig5)

# Boxplot — Annual Income
with col6:
    fig6, ax6 = plt.subplots(figsize=(5, 4.5))
    ax6.boxplot(filtered_df['AMT_INCOME_TOTAL'].dropna(), vert=True, patch_artist=True,
                boxprops=dict(facecolor=colors['box_income']))
    ax6.set_title("Boxplot: Annual Income", fontsize=14, color='white')
    ax6.set_ylabel("Annual Income", fontsize=12, color='white')
    ax6.tick_params(axis='y', colors='white')
    style_axis(ax6)
    st.pyplot(fig6)

col7, col8 = st.columns(2)

# Boxplot — Credit Amount
with col7:
    fig7, ax7 = plt.subplots(figsize=(5, 4.5))
    ax7.boxplot(filtered_df['AMT_CREDIT'].dropna(), vert=True, patch_artist=True,
                boxprops=dict(facecolor=colors['box_credit']))
    ax7.set_title("Boxplot: Credit Amount", fontsize=14, color='white')
    ax7.set_ylabel("Credit Amount", fontsize=12, color='white')
    ax7.tick_params(axis='y', colors='white')
    style_axis(ax7)
    st.pyplot(fig7)

# Countplot — Gender
with col8:
    gender_counts = filtered_df['CODE_GENDER'].value_counts()
    fig8, ax8 = plt.subplots(figsize=(5, 4.5))
    bars = ax8.bar(gender_counts.index, gender_counts.values, color=colors['gender'])
    ax8.set_title("Gender Distribution", fontsize=14, color='white')
    ax8.set_ylabel("Count", fontsize=12, color='white')
    ax8.tick_params(axis='x', colors='white')
    ax8.tick_params(axis='y', colors='white')
    style_axis(ax8)
    ax8.legend(bars, ["Female", "Male", "Other"], fontsize=10)
    st.pyplot(fig8)

col9, col10 = st.columns(2)

# Countplot — Family Status
with col9:
    fam_counts = filtered_df['NAME_FAMILY_STATUS'].value_counts()
    fig9, ax9 = plt.subplots(figsize=(6, 4.5))
    bars = ax9.bar(fam_counts.index, fam_counts.values, color=colors['family'])
    ax9.set_title("Family Status", fontsize=14, color='white')
    ax9.set_ylabel("Count", fontsize=12, color='white')
    ax9.tick_params(axis='x', rotation=45, colors='white')
    ax9.tick_params(axis='y', colors='white')
    style_axis(ax9)
    st.pyplot(fig9)

# Countplot — Education Type
with col10:
    edu_counts = filtered_df['NAME_EDUCATION_TYPE'].value_counts()
    fig10, ax10 = plt.subplots(figsize=(6, 4.5))
    bars = ax10.bar(edu_counts.index, edu_counts.values, color=colors['education'])
    ax10.set_title("Education Type", fontsize=14, color='white')
    ax10.set_ylabel("Count", fontsize=12, color='white')
    ax10.tick_params(axis='x', rotation=90, colors='white')
    ax10.tick_params(axis='y', colors='white')
    style_axis(ax10)
    st.pyplot(fig10)

st.markdown("---")

# ===================== INSIGHTS SECTION =====================
st.subheader("🔍 Key Insights")
st.write("""
- Most applicants repay loans, but a notable minority defaults, showing manageable credit risk.
- Income and credit are highly skewed with outliers; median values better represent typical applicants.
- The age of applicants follows a roughly normal distribution, slightly skewed to the right. The peak appears to be in the 30-50 year range, indicating the bank's primary customer base for loans is working-age adults. There is a sharp drop-off after approximately 60 years, which could reflect bank policy, retirement status affecting credit eligibility, or a natural tendency for older individuals to take on less debt.
- Majority of applicants are standard demographics: male/female, married/single, with secondary or higher education.
""")
    