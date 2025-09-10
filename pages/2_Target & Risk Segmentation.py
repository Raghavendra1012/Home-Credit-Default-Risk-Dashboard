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


st.title("🎯Target & Risk Segmentation")

#=============KPIs==============

total_defaults = df['TARGET'].sum()
default_rate = df['TARGET'].mean()*100
default_by_gender = df.groupby('CODE_GENDER')['TARGET'].mean()*100
default_by_education = df.groupby('NAME_EDUCATION_TYPE')['TARGET'].mean()*100
default_by_family = df.groupby('NAME_FAMILY_STATUS')['TARGET'].mean()*100
default_by_housing = df.groupby('NAME_HOUSING_TYPE')['TARGET'].mean()*100
defaulters = df[df['TARGET'] == 1]
avg_income_defaulters = defaulters['AMT_INCOME_TOTAL'].mean()
avg_credit_defaulters = defaulters['AMT_CREDIT'].mean()
avg_annuity_defaulters = defaulters['AMT_ANNUITY'].mean()
avg_employment_years_defaulters = defaulters['EMPLOYMENT_YEARS'].mean()


col1,col2,col3,col4,col5 = st.columns(5)
col1.metric("Total_Defaults",total_defaults)
col2.metric("Avg_Employment_Years_Defaulters",f"{avg_employment_years_defaulters:.1f} years")
col3.metric("Avg_Income_Defaulters",f"${avg_income_defaulters:,.0f}")
col4.metric("Avg_Credit_Defaulters",f"${avg_credit_defaulters:,.0f}")
col5.metric("Avg_Annuity_Defaulters",f"${avg_annuity_defaulters:,.0f}")

col6,col7,col8,col9,col10 = st.columns(5)
col6.metric("Default_Rate",f"{default_rate:.2f}%")
col7.metric("Default_Rate_by_Gender",f"{default_by_gender.mean():.2f}%")
col8.metric("Default_Rate_by_Education",f"{default_by_education.mean():.2f}%")
col9.metric("Default_Rate_by_Family",f"{default_by_family.mean():.2f}%")
col10.metric("Default_Rate_by_Housing",f"{default_by_housing.mean():.2f}%")


st.markdown("---")

#1. Bar — Counts: Default vs Repaid

col1, col2 = st.columns(2)
with col1:
    target_counts = df['TARGET'].value_counts().sort_index()
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    bars = ax1.bar(["Repaid (0)", "Default (1)"], target_counts.values,
                   color=["#07F10F", "#F31505"])
    ax1.set_title("Counts: Default vs Repaid", fontsize=18)
    ax1.set_ylabel("Count",fontsize=16)
    ax1.grid(True, which='both', axis='y', linestyle='-')
    ax1.tick_params(axis='x', labelsize=14)
    st.pyplot(fig1)

#2. Bar — Default % by Gender
with col2:
    gender_default = df.groupby('CODE_GENDER')['TARGET'].mean() * 100
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    bars = ax2.bar(gender_default.index, gender_default.values,
                   color=["#038DFD", "#FF6600"])
    ax2.set_title("Default % by Gender", fontsize=18)
    ax2.set_ylabel("Default Rate (%)", fontsize=14)
    ax2.grid(True, which='both', axis='y', linestyle='-')
    ax2.tick_params(axis='x', labelsize=14)
    st.pyplot(fig2)

col3, col4 = st.columns(2)

#3. Bar — Default % by Education
with col3:
    edu_default = df.groupby('NAME_EDUCATION_TYPE')['TARGET'].mean().sort_values(ascending=False) * 100
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    bars = ax3.bar(edu_default.index, edu_default.values, color="#D607FA")
    ax3.set_title("Default % by Education", fontsize=18)
    ax3.set_ylabel("Default Rate (%)", fontsize=16)
    ax3.grid(True, which='both', axis='y', linestyle='-')
    ax3.tick_params(axis='x', labelsize=14, rotation=90)
    st.pyplot(fig3)

#4. Bar — Default % by Family Status
with col4:
    fam_default = df.groupby('NAME_FAMILY_STATUS')['TARGET'].mean().sort_values(ascending=False) * 100
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    bars = ax4.bar(fam_default.index, fam_default.values, color="#3C07FA")
    ax4.set_title("Default % by Family Status", fontsize=18)
    ax4.set_ylabel("Default Rate (%)", fontsize=16)
    ax4.grid(True, which='both', axis='y', linestyle='-')
    ax4.tick_params(axis='x', labelsize=14, rotation=90)
    st.pyplot(fig4)

col5, col6 = st.columns(2)

#5. Bar — Default % by Housing Type
with col5:
    housing_default = df.groupby('NAME_HOUSING_TYPE')['TARGET'].mean().sort_values(ascending=False) * 100
    fig5, ax5 = plt.subplots(figsize=(10, 8))
    bars = ax5.bar(housing_default.index, housing_default.values, color="#FA8707")
    ax5.set_title("Default % by Housing Type", fontsize=18)
    ax5.set_ylabel("Default Rate (%)", fontsize=16)
    ax5.grid(True, which='both', axis='y', linestyle='-')
    ax5.tick_params(axis='x', labelsize=14, rotation=90)
    st.pyplot(fig5)

#Boxplot — Income by Target
with col6:
    fig6, ax6 = plt.subplots(figsize=(14, 12))
    ax6.boxplot([df[df['TARGET'] == 0]['AMT_INCOME_TOTAL'],
                  df[df['TARGET'] == 1]['AMT_INCOME_TOTAL']],
                 labels=['Repaid (0)', 'Default (1)'],
                 patch_artist=True,
                 boxprops=dict(facecolor="#06F30E"),
                 medianprops=dict(color='black'))
    ax6.set_title("Income Distribution by Target", fontsize=24)
    ax6.set_ylabel("Annual Income", fontsize=20)
    ax6.tick_params(axis='x', labelsize=17)
    st.pyplot(fig6)

col7, col8 = st.columns(2)

# Boxplot — Credit by Target
with col7:
    fig7, ax7 = plt.subplots(figsize=(14, 12))
    ax7.boxplot([df[df['TARGET'] == 0]['AMT_CREDIT'],
                  df[df['TARGET'] == 1]['AMT_CREDIT']],
                 labels=['Repaid (0)', 'Default (1)'],
                 patch_artist=True,
                 boxprops=dict(facecolor="#1606F3"),
                 medianprops=dict(color='yellow'))
    ax7.set_title("Credit Amount Distribution by Target", fontsize=24)
    ax7.set_ylabel("Credit Amount", fontsize=20)
    ax7.tick_params(axis='x', labelsize=17)
    st.pyplot(fig7)

# Violin Plot — Age vs Target
with col8:
    fig8, ax8 = plt.subplots(figsize=(14, 12))
    sns.violinplot(x='TARGET', y='AGE_YEARS', data=df, ax=ax8, palette=["#1606F3", "#F306A8"])
    ax8.set_title("Age Distribution by Target", fontsize=24)
    ax8.set_ylabel("Age", fontsize=22)
    ax8.set_xlabel("Target", fontsize=20)
    ax8.set_xticklabels(["Repaid (0)", "Default (1)"])
    ax8.tick_params(axis='x', labelsize=17)
    st.pyplot(fig8)

col9, col10 = st.columns(2)

# Stacked Histogram — Employment Years by Target
with col9:
    bins = [0, 1, 3, 5, 10, 20, 30, 40, 50]  # Define bins for employment years
    fig9, ax9 = plt.subplots(figsize=(14, 12))
    
    ax9.hist([df[df['TARGET'] == 0]['EMPLOYMENT_YEARS'],
              df[df['TARGET'] == 1]['EMPLOYMENT_YEARS']],
             bins=bins,
             stacked=True,
             color=['#1606F3', "#F32606"],
             label=['Repaid (0)', 'Default (1)'])
    
    ax9.set_xlabel("Employment Years", fontsize=20)
    ax9.set_ylabel("Count", fontsize=20)
    ax9.grid(True, which='both', axis='y', linestyle='-')
    ax9.set_title("Employment Years Distribution by Target", fontsize=24)
    ax9.legend(fontsize=16)
    st.pyplot(fig9)

# Stacked Bar Chart — Name Contract Type by Target
with col10:
    contract_counts = df.groupby(['NAME_CONTRACT_TYPE', 'TARGET']).size().unstack(fill_value=0)
    
    fig10, ax10 = plt.subplots(figsize=(10, 8))
    contract_counts.plot(kind='bar', stacked=True, color=["#0EF306", '#F306A8'], ax=ax10)
    
    ax10.set_xlabel("Contract Type", fontsize=20)
    ax10.set_ylabel("Count", fontsize=20)
    ax10.grid(True, which='both', axis='y', linestyle='-')
    ax10.set_title("Contract Type vs Target", fontsize=24)
    ax10.legend(['Repaid (0)', 'Default (1)'], fontsize=16)
    ax10.tick_params(axis='x', rotation=0)
    
    st.pyplot(fig10)

st.markdown("---")

# ===================== INSIGHTS SECTION =====================
st.subheader("🔍 Key Insights")
st.write("""
- Default rates are higher among younger applicants, indicating a potential risk factor.
- Applicants with higher income and credit amounts tend to have lower default rates.
- Younger applicants and those with fewer employment years tend to default more.
- Certain contract types (e.g., cash loans) show higher default rates compared to others (e.g., revolving loans).
""")