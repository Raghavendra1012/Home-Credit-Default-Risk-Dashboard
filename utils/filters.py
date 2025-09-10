import streamlit as st
import pandas as pd

def global_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Display global filters and apply them to the DataFrame.

    Args:
        df (pd.DataFrame): The original DataFrame with necessary columns.

    Returns:
        tuple: (original DataFrame, filtered DataFrame)
    """

    st.sidebar.header("Global Filters")

    # Initialize session state for persistence across pages
    if 'filters_initialized' not in st.session_state:
        st.session_state.filters_initialized = True
        st.session_state.age_range = (int(df['AGE_YEARS'].min()), int(df['AGE_YEARS'].max()))
        st.session_state.income_bracket = []

    # Gender filter
    gender_options = df['CODE_GENDER'].dropna().unique()
    gender = st.sidebar.multiselect("Gender", gender_options, key='gender_filter')

    # Education filter
    education_options = df['NAME_EDUCATION_TYPE'].dropna().unique()
    education = st.sidebar.multiselect("Education", education_options, key='education_filter')

    # Family Status filter
    family_status_options = df['NAME_FAMILY_STATUS'].dropna().unique()
    family_status = st.sidebar.multiselect("Family Status", family_status_options, key='family_status_filter')

    # Housing Type filter
    housing_options = df['NAME_HOUSING_TYPE'].dropna().unique()
    housing = st.sidebar.multiselect("Housing Type", housing_options, key='housing_filter')

    # Age Range filter
    min_age = int(df['AGE_YEARS'].min())
    max_age = int(df['AGE_YEARS'].max())
    age_range = st.sidebar.slider("Age Range", min_value=min_age, max_value=max_age,
                                  value=st.session_state.age_range, key='age_range_filter')
    st.session_state.age_range = age_range

   
    # Income Bracket filter
    max_income = df['AMT_INCOME_TOTAL'].max()
    # Ensure the last bin is bigger than the previous one
    if max_income <= 1000000:
        max_income = 1000001  # Slightly larger than the previous boundary

    bins = [0, 200000, 500000, 1000000, max_income]
    labels = ['Low (0–2L)', 'Medium (2–5L)', 'High (5–10L)', 'Very High (10L+)']

    df['income_bracket'] = pd.cut(df['AMT_INCOME_TOTAL'], bins=bins, labels=labels)

    income_options = labels
    income_bracket = st.sidebar.multiselect("Income Bracket", income_options,
                                            default=st.session_state.income_bracket, key='income_bracket_filter')
    st.session_state.income_bracket = income_bracket

    # Apply filters
    filtered_df = df[
        (df['CODE_GENDER'].isin(gender) if gender else True) &
        (df['NAME_EDUCATION_TYPE'].isin(education) if education else True) &
        (df['NAME_FAMILY_STATUS'].isin(family_status) if family_status else True) &
        (df['NAME_HOUSING_TYPE'].isin(housing) if housing else True) &
        (df['AGE_YEARS'].between(age_range[0], age_range[1])) &
        (df['income_bracket'].isin(income_bracket) if income_bracket else True)
    ]

    return filtered_df