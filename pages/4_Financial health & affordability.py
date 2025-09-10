import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from utils.load_data import load_and_preprocess
from utils.filters import global_filters
import pandas as pd
import numpy as np

df = load_and_preprocess()

# Apply global filters
filtered_df = global_filters(df)

# Show filtered data
st.write("Filtered Data", filtered_df.head(10))

st.title("💰Financial Health & Affordability")

#=============KPIs==============

avg_annual_income = df['AMT_INCOME_TOTAL'].mean()
median_annual_income = df['AMT_INCOME_TOTAL'].median()
avg_credit_amount = df['AMT_CREDIT'].mean()
avg_annuity_amount = df['AMT_ANNUITY'].mean()
avg_goods_price = df['AMT_GOODS_PRICE'].mean()
avg_debt_to_income_ratio = (df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']).mean() * 100
avg_loan_to_income_ratio = (df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']).mean() * 100
income_gap = df[df['TARGET'] == 0]['AMT_INCOME_TOTAL'].mean() - df[df['TARGET'] == 1]['AMT_INCOME_TOTAL'].mean()
credit_gap = df[df['TARGET'] == 0]['AMT_CREDIT'].mean() - df[df['TARGET'] == 1]['AMT_CREDIT'].mean()
pct_high_credit = (df['AMT_CREDIT'] > 1000000).mean() * 100

col1,col2,col3,col4,col5 = st.columns(5)

col1.metric("Avg Annual Income", f"${avg_annual_income:,.2f}")
col2.metric("Median Annual Income", f"${median_annual_income:,.2f}")
col3.metric("Avg Credit Amount", f"${avg_credit_amount:,.2f}")
col4.metric("Avg Annuity Amount", f"${avg_annuity_amount:,.2f}")
col5.metric("Avg Goods Price", f"${avg_goods_price:,.2f}")

col6,col7,col8,col9,col10 = st.columns(5)
col6.metric("Avg Debt-to-Income Ratio", f"{avg_debt_to_income_ratio:.2f}%")
col7.metric("Avg Loan-to-Income Ratio", f"{avg_loan_to_income_ratio:.2f}%")
col8.metric("Income Gap (Non-Def vs Def)", f"${income_gap:,.2f}")
col9.metric("Credit Gap (Non-Def vs Def)", f"${credit_gap:,.2f}")
col10.metric("High Credit (> $1M) %", f"{pct_high_credit:.2f}%")

st.markdown("---")

col1, col2 = st.columns(2)

# Histogram: Income distribution (all)

with col1:
    fig1, ax1 = plt.subplots(figsize=(10,6))
    ax1.hist(df['AMT_INCOME_TOTAL'], bins=30, color="#1606F3", alpha=0.7, edgecolor="black")
    ax1.set_title("Annual Income Distribution (All Applicants)", fontsize=20)
    ax1.set_xlabel("Annual Income ($)", fontsize=16)
    ax1.set_ylabel("Count", fontsize=16)
    ax1.grid(True, which='both', axis='y', linestyle='-')
    st.pyplot(fig1)

# Histogram: Credit Distribution (all)

with col2:
    fig2, ax2 = plt.subplots(figsize=(10,6))
    ax2.hist(df['AMT_CREDIT'], bins=30, color="#F31616", alpha=0.7, edgecolor="black")
    ax2.set_title("Credit Amount Distribution (All Applicants)", fontsize=20)
    ax2.set_xlabel("Credit Amount ($)", fontsize=16)
    ax2.set_ylabel("Count", fontsize=16)
    ax2.grid(True, which='both', axis='y', linestyle='-')
    st.pyplot(fig2)

col3, col4 = st.columns(2)

# Histogram: Annuity Distribution (all)

with col3:
    fig3, ax3 = plt.subplots(figsize=(10,6))
    ax3.hist(df['AMT_ANNUITY'], bins=30, color="#16F316", alpha=0.7, edgecolor="black")
    ax3.set_title("Annuity Amount Distribution (All Applicants)", fontsize=20)
    ax3.set_xlabel("Annuity Amount ($)", fontsize=16)
    ax3.set_ylabel("Count", fontsize=16)
    ax3.grid(True, which='both', axis='y', linestyle='-')
    st.pyplot(fig3)

# Scatter: Income vs Credit

with col4:
    fig4, ax4 = plt.subplots(figsize=(10,6))
    scatter = ax4.scatter(df['AMT_INCOME_TOTAL'], df['AMT_CREDIT'], c=df['TARGET'], cmap="coolwarm", alpha=0.7)
    ax4.set_title("Income vs Credit Amount (Colored by Target)", fontsize=20)
    ax4.set_xlabel("Annual Income ($)", fontsize=16)
    ax4.set_ylabel("Credit Amount ($)", fontsize=16)
    fig4.colorbar(scatter, ax=ax4, label="Target")
    st.pyplot(fig4)

col5, col6 = st.columns(2)

# Scatter: Income vs Annuity

with col5:
    fig5, ax5 = plt.subplots(figsize=(10,6))
    scatter = ax5.scatter(df['AMT_INCOME_TOTAL'], df['AMT_ANNUITY'], alpha=0.7)
    ax5.set_title("Income vs Annuity Amount (Colored by Target)", fontsize=20)
    ax5.set_xlabel("Annual Income ($)", fontsize=16)
    ax5.set_ylabel("Annuity Amount ($)", fontsize=16)
    fig5.colorbar(scatter, ax=ax5, label="Target")
    st.pyplot(fig5)

# Boxplot: Credit by Target

with col6:
    fig6, ax6 = plt.subplots(figsize=(10,6))
    sns.boxplot(x='TARGET', y='AMT_CREDIT', data=df, ax=ax6)
    ax6.set_title("Credit Amount by Target", fontsize=20)
    ax6.set_xlabel("Target", fontsize=16)
    ax6.set_ylabel("Credit Amount ($)", fontsize=16)
    ax6.grid(True, which='both', axis='y', linestyle='-')
    st.pyplot(fig6)

col7, col8 = st.columns(2)

# Boxplot: Income by Target

with col7:
    fig7, ax7 = plt.subplots(figsize=(10,6))
    sns.boxplot(x='TARGET', y='AMT_INCOME_TOTAL', data=df, ax=ax7)
    ax7.set_title("Income Amount by Target", fontsize=20)
    ax7.set_xlabel("Target", fontsize=16)
    ax7.set_ylabel("Income Amount ($)", fontsize=16)
    ax7.grid(True, which='both', axis='y', linestyle='-')
    st.pyplot(fig7)

# KDE/Density: Joint Income Credit

with col8:
    fig8, ax8 = plt.subplots(figsize=(10,6))
    sns.kdeplot(x='AMT_INCOME_TOTAL', y='AMT_CREDIT', data=df, ax=ax8, cmap="Blues", fill=True)
    ax8.set_title("Joint Income and Credit Density", fontsize=20)
    ax8.set_xlabel("Annual Income ($)", fontsize=16)
    ax8.set_ylabel("Credit Amount ($)", fontsize=16)
    st.pyplot(fig8)

col9, col10 = st.columns(2)

# Barplot: Income Bracket vs Default Rate

with col9:
    income_default_df = df.groupby('INCOME_BRACKET').agg(default_rate=('TARGET', 'mean')).reset_index()
    income_default_df['default_rate'] = income_default_df['default_rate'] * 100
    fig9, ax9 = plt.subplots(figsize=(10,6))
    sns.barplot(x='INCOME_BRACKET', y='default_rate', data=income_default_df, ax=ax9)
    ax9.set_title("Income Bracket vs Default Rate", fontsize=20)
    ax9.set_xlabel("Income Bracket", fontsize=16)
    ax9.set_ylabel("Default Rate (%)", fontsize=16)
    ax9.grid(True, which='both', axis='y', linestyle='-')
    st.pyplot(fig9)

# Heatmap — Financial variable correlations (Income, Credit, Annuity, DTI, LTI, TARGET)

with col10:
    df['debt_to_income_ratio'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['loan_to_income_ratio'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    financial_corr_data = df[['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'debt_to_income_ratio', 'loan_to_income_ratio', 'TARGET']]
    financial_corr = financial_corr_data.corr()

    fig10, ax10 = plt.subplots(figsize=(8, 6))
    sns.heatmap(financial_corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax10)
    ax10.set_title("Financial Variables Correlation Heatmap", fontsize=20)
    st.pyplot(fig10)

st.markdown("---")

# ===================== INSIGHTS SECTION =====================
st.subheader("🔍 Key Insights")
st.write("""
- **Income & Credit**: Higher income applicants generally receive higher credit amounts, but there is significant variability.
- Higher credit and annuity amounts tend to be associated with higher-income applicants, though some defaulters request large credits.
- Default rates are higher in lower income brackets, as shown in the income vs default barplot.
- Financial ratios (debt-to-income and loan-to-income) are positively correlated with credit and annuity, and moderately linked to default risk.
""")