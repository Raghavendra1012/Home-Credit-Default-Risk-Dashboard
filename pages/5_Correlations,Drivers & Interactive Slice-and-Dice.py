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

st.title("🔗Correlation, Drivers & Interactive Slice-and-Dice")
st.write("Explore numeric correlations to TARGET and test candidate rules. All visuals respond to the global filters.")


#=============KPIs==============

# Keep only numeric columns for correlation
numeric_df = df.select_dtypes(include=['number'])

# Calculate correlations with TARGET
corr_matrix = numeric_df.corr()
target_corr = corr_matrix['TARGET'].sort_values(ascending=False)

# Top Positive Correlations
top_pos = target_corr.drop('TARGET', errors='ignore').head(5)

# Top Negative Correlations (take most negative by sorting ascending)
top_neg = target_corr.drop('TARGET', errors='ignore').sort_values(ascending=True).head(5)

# Most correlated with Income
income_corr = corr_matrix['AMT_INCOME_TOTAL'].drop('AMT_INCOME_TOTAL', errors='ignore')
most_income = income_corr.abs().idxmax() if not income_corr.empty else 'N/A'

# Most correlated with Credit
credit_corr = corr_matrix['AMT_CREDIT'].drop('AMT_CREDIT', errors='ignore')
most_credit = credit_corr.abs().idxmax() if not credit_corr.empty else 'N/A'

# Additional correlations with fallback if column missing
corr_income_credit = corr_matrix.loc['AMT_INCOME_TOTAL', 'AMT_CREDIT'] if 'AMT_INCOME_TOTAL' in corr_matrix.index and 'AMT_CREDIT' in corr_matrix.columns else 0
corr_employment_target = corr_matrix.loc['EMPLOYMENT_YEARS', 'TARGET'] if 'EMPLOYMENT_YEARS' in corr_matrix.index and 'TARGET' in corr_matrix.columns else 0
corr_age_target = corr_matrix.loc['AGE_YEARS', 'TARGET'] if 'AGE_YEARS' in corr_matrix.index and 'TARGET' in corr_matrix.columns else 0
corr_family_target = corr_matrix.loc['CNT_FAM_MEMBERS', 'TARGET'] if 'CNT_FAM_MEMBERS' in corr_matrix.index and 'TARGET' in corr_matrix.columns else 0

# Layout for Streamlit
st.subheader("KPIs — Correlation & drivers")
col1, col2 = st.columns(2)

with col1:
    st.write("### Top POS(+) Corr with TARGET")
    for feature, value in top_pos.items():
        st.write(f"{feature} ({value:.2f})")
        
with col2:
    st.write("### Top NEG(−) Corr with TARGET")
    for feature, value in top_neg.items():
        st.write(f"{feature} ({value:.2f})")

col3, col4 = st.columns(2)

with col3:
    st.write("### Most correlated with Income")
    st.write(f"**{most_income}**")
    st.write("Corr(Income, Credit)")
    st.write(f"{corr_income_credit:.2f}")
    st.write("Corr(EmploymentYears, TARGET)")
    st.write(f"{corr_employment_target:.2f}")

with col4:
    st.write("### Most correlated with Credit")
    st.write(f"**{most_credit}**")
    st.write("Corr(Age, TARGET)")
    st.write(f"{corr_age_target:.2f}")
    st.write("Corr(FamilySize, TARGET)")
    st.write(f"{corr_family_target:.2f}")

st.markdown("---")

c1,c2 = st.columns(2)

with c1:
    st.subheader("Heatmap — Correlation (selected numerics)")

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Multiselect for columns
    selected_columns = st.multiselect("Select numeric columns", numeric_cols, default=numeric_cols[:10])
    
    if selected_columns:
        corr_matrix = df[selected_columns].corr()

        # Plotting heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, ax=ax)
        ax.set_title("Correlation Heatmap", fontsize=14)

        st.pyplot(fig)
    else:
        st.write("Please select at least one numeric column.")


with c2:
    st.subheader("|Correlation| vs TARGET (top features)")
    # Select top N features
    N = 10
    top_features = target_corr.sort_values(ascending=False).head(N)
    st.write(f"### Top {N} Features by |Correlation| with TARGET")
    fig, ax = plt.subplots(figsize=(8, 5))
    top_features.plot(kind='bar', ax=ax)
    ax.set_xlabel("Features")
    ax.set_ylabel("|Correlation|")
    ax.set_title(f"Top {N} Correlated Features")
    ax.grid(True, which='both', axis='y', linestyle='-')
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

c3, c4 = st.columns(2)

with c3:
    st.write("### Scatter: Age vs Credit")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(
        data=df,
        x='AGE_YEARS',
        y='AMT_CREDIT',
        hue='TARGET',
        palette='coolwarm',
        alpha=0.6,
        ax=ax
    )
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Credit Amount")
    ax.set_title("Age vs Credit by TARGET")
    plt.tight_layout()
    st.pyplot(fig)

with c4:
    st.write("### Scatter: Age vs Income")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(
        data=df,
        x='AGE_YEARS',
        y='AMT_INCOME_TOTAL',
        hue='TARGET',
        palette='coolwarm',
        alpha=0.6,
        ax=ax
    )
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Income")
    ax.set_title("Age vs Income by TARGET")
    plt.tight_layout()
    st.pyplot(fig)

c5, c6 = st.columns(2)

with c5:
    st.write("### Scatter: Employment Years vs TARGET (with jitter)")
    fig, ax = plt.subplots(figsize=(6, 5))
    # Add jitter by slightly randomizing the employment years
    jitter = np.random.uniform(-0.3, 0.3, size=len(df))
    sns.scatterplot(
        x=df['EMPLOYMENT_YEARS'] + jitter,
        y=df['TARGET'],
        alpha=0.3
    )
    ax.set_xlabel("Years Employed")
    ax.set_ylabel("TARGET")
    ax.set_title("Employment Years vs TARGET with jitter")
    plt.tight_layout()
    st.pyplot(fig)

with c6:
    st.write("### Boxplot: Credit by Education")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(
        data=df,
        x='NAME_EDUCATION_TYPE',
        y='AMT_CREDIT'
    )
    ax.set_xlabel("Education Level")
    ax.set_ylabel("Credit Amount")
    ax.set_title("Credit by Education")
    ax.grid(True, which='both', axis='y', linestyle='-')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

c7, c8 = st.columns(2)

with c7:
    st.write("### Boxplot: Income by Family Status")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(
        data=df,
        x='NAME_FAMILY_STATUS',
        y='AMT_INCOME_TOTAL'
    )
    ax.set_xlabel("Family Status")
    ax.set_ylabel("Income")
    ax.set_title("Income by Family Status")
    ax.grid(True, which='both', axis='y', linestyle='-')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

with c8:
    st.write("### Pair Plot: Income, Credit, Annuity, TARGET")
    # Select the relevant columns
    pair_df = df[['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'TARGET']]
    # Optional: sample for performance if dataset is very large
    if len(pair_df) > 1000:
        pair_df = pair_df.sample(1000, random_state=42)
    sns.pairplot(pair_df, hue='TARGET', diag_kind='kde', corner=True)
    st.pyplot(plt.gcf())

c9,c10 = st.columns(2)

with c9:
    st.write("### Default Rate by Gender")
    gender_rate = filtered_df.groupby('CODE_GENDER')['TARGET'].mean().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=gender_rate.index, y=gender_rate.values, palette='dark', ax=ax)
    ax.set_ylabel("Default Rate")
    ax.set_xlabel("Gender")
    ax.set_ylim(0, 0.2)
    ax.set_title("Default Rate by Gender")
    ax.grid(True, which='both', axis='y', linestyle='-')
    st.pyplot(fig)

with c10:
    st.write("### Default Rate by Education")
    edu_rate = filtered_df.groupby('NAME_EDUCATION_TYPE')['TARGET'].mean().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=edu_rate.index, y=edu_rate.values, palette='dark', ax=ax)
    ax.set_ylabel("Default Rate")
    ax.set_xlabel("Education Level")
    ax.set_ylim(0, 0.2)
    ax.set_title("Default Rate by Education")
    ax.grid(True, which='both', axis='y', linestyle='-')
    plt.xticks(rotation=90, ha='right')
    st.pyplot(fig)

st.markdown("---")

# ===================== INSIGHTS SECTION =====================
st.subheader("🔍 Key Insights")
st.write("""
- **Age & Income**: Older applicants tend to have higher incomes, but this relationship is not very strong.
- **Employment**: Longer employment duration is associated with higher income and credit amounts.
- **Education**: Higher education levels correlate with better financial health indicators.
- **Family Status**: Married applicants generally have higher incomes and lower default rates.
- **Default Rates**: Certain demographic groups (e.g., younger, less educated) exhibit higher default rates.
""")
