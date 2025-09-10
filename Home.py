import streamlit as st
from utils.load_data import load_and_preprocess

# Page config
st.set_page_config(page_title="Home Credit Dashboard", page_icon="🏦", layout="wide")

# Title
st.title("🏦 Home Credit Default Risk Dashboard")

# Intro markdown
st.markdown("""
Welcome to the Home Credit Default Risk Dashboard built with Streamlit.  
This dashboard helps in exploring the application_train.csv dataset to understand
risk factors associated with loan defaults.  

Use the sidebar to navigate between different analysis modules:
- 📊Overview & Data Quality
- 🎯Target & Risk Segmentation
- 👨‍👩‍👧‍👦Demographics & Household Profile
- 💰Financial Health & Affordability
- 🔗Correlations,Drivers & Interactive Slice-and-Dice
""")

st.markdown("---")
st.subheader("Upload / Use Default Dataset")

# File uploader
uploaded_file = st.file_uploader("Upload your Home Credit CSV file (application_train.csv)", type=["csv"])
if uploaded_file:
    df = load_and_preprocess(uploaded_file)
else:
    df = load_and_preprocess()  # Load default application_train.csv from utils

# Show sample data
st.dataframe(df.head(100))

df.shape