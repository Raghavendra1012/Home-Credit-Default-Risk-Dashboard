import pandas as pd
import numpy as np

def load_and_preprocess(file_path="application_train.csv"):
    df = pd.read_csv(file_path)

    # ---- Feature engineering ----
    df['AGE_YEARS'] = -(df['DAYS_BIRTH'] / 365.25).astype(int)
    df['EMPLOYMENT_YEARS'] = -(df['DAYS_EMPLOYED'] / 365.25)
    df['EMPLOYMENT_YEARS'] = df['EMPLOYMENT_YEARS'].clip(lower=0, upper=60)  # handle "not employed" codes

    df['DTI'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['LOAN_TO_INCOME'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_TO_CREDIT'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    # ---- Missing values ----
    missing = df.isnull().mean() * 100
    cols_to_drop = missing[missing > 60].index
    df = df.drop(columns=cols_to_drop)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # ---- Standardize rare categories ----
    for col in df.select_dtypes(include="object").columns:
        freqs = df[col].value_counts(normalize=True)
        rare = freqs[freqs < 0.01].index
        df[col] = df[col].replace(rare, "Other")

    # ---- Outlier handling (winsorize 1%) ----
    for col in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']:
        lower, upper = df[col].quantile([0.01, 0.99])
        df[col] = np.clip(df[col], lower, upper)

    # ---- Income brackets ----
    q1, q2, q3 = df['AMT_INCOME_TOTAL'].quantile([0.25, 0.5, 0.75])
    def income_bracket(x):
        if x <= q1: return "Low"
        elif x <= q3: return "Mid"
        else: return "High"
    df['INCOME_BRACKET'] = df['AMT_INCOME_TOTAL'].apply(income_bracket)

    return df

