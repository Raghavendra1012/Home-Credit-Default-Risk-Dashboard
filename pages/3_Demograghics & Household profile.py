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

st.title("👨‍👩‍👧‍👦Demographics & Household Profile")

#=============KPIs==============

gender_counts = df['CODE_GENDER'].value_counts(normalize=True) * 100
male_pct = gender_counts.get('M', 0)
female_pct = gender_counts.get('F', 0)
avg_age_defaulters = df[df['TARGET'] == 1]['AGE_YEARS'].mean()
avg_age_non_defaulters = df[df['TARGET'] == 0]['AGE_YEARS'].mean()
pct_with_children = (df['CNT_CHILDREN'] > 0).mean() * 100
avg_family_size = df['CNT_FAM_MEMBERS'].mean()
pct_married_vs_single = df['NAME_FAMILY_STATUS'].value_counts(normalize=True) * 100
married_pct = pct_married_vs_single.get('Married', 0)
single_pct = pct_married_vs_single.get('Single / not married', 0)
pct_high_education = (df['NAME_EDUCATION_TYPE'] == 'Higher education').mean() * 100
pct_living_with_parents = (df['NAME_HOUSING_TYPE'] == 'With parents').mean() * 100
pct_currently_employed = (df['EMPLOYMENT_YEARS'] > 0).mean() * 100
avg_employment_years = df['EMPLOYMENT_YEARS'].mean()

col1,col2,col3,col4,col5,col6 = st.columns(6)
col1.metric("Male %", f"{male_pct:.2f}%")
col2.metric("Female %", f"{female_pct:.2f}%")
col3.metric("Avg Age (Defaulters)", f"{avg_age_defaulters:.2f} years")
col4.metric("Avg Age (Non-Defaulters)", f"{avg_age_non_defaulters:.2f} years")
col5.metric("With Children %", f"{pct_with_children:.2f}%")
col6.metric("Avg Family Size", f"{avg_family_size:.2f}")

col7,col8,col9,col10,col11,col12 = st.columns(6)
col7.metric("Married %", f"{married_pct:.2f}%")
col8.metric("Single %", f"{single_pct:.2f}%")
col9.metric("Higher Education %", f"{pct_high_education:.2f}%")
col10.metric("Living with Parents %", f"{pct_living_with_parents:.2f}%")
col11.metric("Currently Employed %", f"{pct_currently_employed:.2f}%")
col12.metric("Avg Employment Years", f"{avg_employment_years:.2f}")

st.markdown("---")


col1, col2 = st.columns(2)

# Histogram: Age distribution (all)

with col1:
    fig1, ax1 = plt.subplots(figsize=(10,6))
    ax1.hist(df['AGE_YEARS'], bins=30, color="#1606F3", alpha=0.7, edgecolor="black")
    ax1.set_title("Age Distribution (All Applicants)", fontsize=20)
    ax1.set_xlabel("Age (years)", fontsize=16)
    ax1.set_ylabel("Count", fontsize=16)
    ax1.grid(True, which='both', axis='y', linestyle='-')
    st.pyplot(fig1)

# Histogram: Age distribution by Target (overlay)

with col2:
    fig2, ax2 = plt.subplots(figsize=(10,6))
    ax2.hist(df[df['TARGET'] == 0]['AGE_YEARS'], bins=30, color="#04F40C", alpha=0.7, label='Non-Defaulters', edgecolor="black")
    ax2.hist(df[df['TARGET'] == 1]['AGE_YEARS'], bins=30, color="#F40404", alpha=0.7, label='Defaulters', edgecolor="black")
    ax2.set_title("Age Distribution by Target", fontsize=20)
    ax2.set_xlabel("Age (years)", fontsize=16)
    ax2.set_ylabel("Count", fontsize=16)
    ax2.legend(fontsize=14)
    ax2.grid(True, which='both', axis='y', linestyle='-')
    st.pyplot(fig2)

col3, col4 = st.columns(2)

# Gender distribution

with col3:
    fig3, ax3 = plt.subplots(figsize=(10,6))
    sns.countplot(data=df, x='CODE_GENDER', ax=ax3, palette=["#1606F3", "#F306A8", "#06F3F2"])
    ax3.set_title("Gender Distribution", fontsize=20)
    ax3.set_xlabel("Gender", fontsize=16)
    ax3.set_ylabel("Count", fontsize=16)
    ax3.grid(True, which='both', axis='y', linestyle='-')
    ax3.legend(labels=["Male", "Female", "Other"], fontsize=14)
    st.pyplot(fig3)

# Family status distribution

with col4:
    fig4, ax4 = plt.subplots(figsize=(10,6))
    sns.countplot(data=df, x='NAME_FAMILY_STATUS', ax=ax4, palette="Set2", order=df['NAME_FAMILY_STATUS'].value_counts().index)
    ax4.set_title("Family Status Distribution", fontsize=20)
    ax4.set_xlabel("Family Status", fontsize=16)
    ax4.set_ylabel("Count", fontsize=16)
    ax4.tick_params(axis='x', rotation=90)
    ax4.grid(True, which='both', axis='y', linestyle='-')
    st.pyplot(fig4)

col5, col6 = st.columns(2)

# Education Distribution

with col5:
    fig5, ax5 = plt.subplots(figsize=(10,6))
    sns.countplot(data=df, x='NAME_EDUCATION_TYPE', ax=ax5, palette="Set2", order=df['NAME_EDUCATION_TYPE'].value_counts().index)
    ax5.set_title("Education Distribution", fontsize=20)
    ax5.set_xlabel("Education Level", fontsize=16)
    ax5.set_ylabel("Count", fontsize=16)
    ax5.tick_params(axis='x', rotation=90)
    ax5.grid(True, which='both', axis='y', linestyle='-')
    st.pyplot(fig5)

## Occupation distribution (top 10)

with col6:
    top10_occup = df['OCCUPATION_TYPE'].value_counts().nlargest(10)

    fig6, ax6 = plt.subplots(figsize=(10,6))
    top10_occup.plot(kind='bar', color="#1606F3", ax=ax6)
    ax6.set_title("Top 10 Occupation Types", fontsize=18)
    ax6.set_xlabel("Occupation", fontsize=14)
    ax6.set_ylabel("Count", fontsize=14)
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, which='both', axis='y', linestyle='-')
    st.pyplot(fig6)

col7, col8 = st.columns(2)

# Pie - Housing type distribution

with col7:
    housing_counts = df['NAME_HOUSING_TYPE'].value_counts()
    fig7, ax7 = plt.subplots(figsize=(8,8))
    ax7.pie(housing_counts, labels=housing_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set2", len(housing_counts)))
    ax7.set_title("Housing Type Distribution", fontsize=20)
    st.pyplot(fig7)

#Countplot — CNT_CHILDREN
with col8:
    fig8, ax8 = plt.subplots(figsize=(10,6))
    sns.countplot(data=df, x='CNT_CHILDREN', ax=ax8, palette="Set1")
    ax8.set_title("Number of Children Distribution", fontsize=20)
    ax8.set_xlabel("Number of Children", fontsize=16)
    ax8.set_ylabel("Count", fontsize=16)
    ax8.grid(True, which='both', axis='y', linestyle='-')
    st.pyplot(fig8)

col9, col10 = st.columns(2)

#Boxplot — Age vs Target

with col9:
    fig9, ax9 = plt.subplots(figsize=(10,6))
    sns.boxplot(data=df, x='TARGET', y='AGE_YEARS', ax=ax9, palette=["#0424F4", "#F40404"])
    ax9.set_title("Age vs Target", fontsize=20)
    ax9.set_xlabel("Target", fontsize=16)
    ax9.set_ylabel("Age (years)", fontsize=16)
    st.pyplot(fig9)

# Heatmap — Corr(Age, Children, Family Size, TARGET)

with col10:
    corr_data = df[['AGE_YEARS', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'TARGET']]
    corr = corr_data.corr()

    fig10, ax10 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax10)
    ax10.set_title("Correlation Heatmap", fontsize=20)
    st.pyplot(fig10)

st.markdown("---")

# ===================== INSIGHTS SECTION =====================
st.subheader("🔍 Key Insights")
st.write("""
- Most applicants are middle-aged, with defaulters slightly younger on average.
- Majority of applicants are male, married/single, with secondary or higher education.
- A significant portion of applicants have children and live in rented apartments.
- Age, family size, and number of children show weak correlations with default (TARGET).
""")