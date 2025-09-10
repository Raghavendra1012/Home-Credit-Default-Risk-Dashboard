
# 🏦 Home Credit Default Risk Dashboard

## 📊 Project Overview

A comprehensive **Streamlit dashboard** for analyzing loan application data and predicting default risk using the **Home Credit** dataset. This interactive tool helps financial institutions and data analysts explore key financial and behavioral metrics, identify high-risk applicants, and uncover patterns affecting creditworthiness through **visual analytics** and **machine learning insights**.

---

## 🚀 Key Features

✅ **5 interactive pages** offering multiple analytical perspectives  
✅ **50+ KPIs** tracking critical risk metrics in real time  
✅ Fully **responsive to global filters**, enabling custom data exploration  
✅ **Dark-themed, aesthetically pleasing** visualizations optimized for readability  
✅ Dynamic correlation analysis with **heatmaps, scatterplots, boxplots, and pair plots**  
✅ Integration of demographic, financial, and behavioral features

---

## 📈 Key Metrics Tracked

✔ Default rates across demographic segments (gender, education, family status)  
✔ Income-to-loan ratios and financial capacity indicators  
✔ Employment history patterns and job stability analysis  
✔ Family size, structure, and household-related risk factors  
✔ Financial behavior indicators derived from credit, annuity, and income data  
✔ Correlations between features and target outcomes to aid feature selection

---

## 🛠 Environment & Dependencies

**Python 3.10+** is recommended. Install the required libraries:

```bash
pip install streamlit pandas numpy matplotlib seaborn
```

---

## 📂 Data Requirements

**File needed:** `application_train.csv`  
**Default location:** `data/application_train.csv`

You can adjust the file path in `utils/load_data.py` if required.

---

## ⚙ Preprocessing

The `utils` directory includes preprocessing scripts:

- **load_data.py** – loads and cleans the dataset, handles missing values, and applies necessary transformations.  
- **filters.py** – defines global filters that dynamically affect all dashboard components.

---

## 🚀 How to Run

1. Clone the repository:

    ```bash
    git clone <your-repo-url>
    cd <your-repo-folder>
    ```

2. Place the dataset (`application_train.csv`) in the `data/` folder.

3. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

4. Access the dashboard in your browser at the provided localhost URL.

---

## 📌 Notes

- The dashboard is optimized for performance and readability with compact figures.  
- Large datasets are sampled where necessary to ensure smooth interactions.  
- Outliers in boxplots are styled in white for enhanced visibility on dark backgrounds.  
- All metrics, plots, and KPIs automatically adjust to applied filters for in-depth analysis.

---

## 📬 Contributions

Feel free to fork the project, raise issues, or submit pull requests to improve features, fix bugs, or enhance the user experience!

---

Check out my Dashboard - https://home-credit-default-risk-dashboard-e2ipqcamisxuin2qtuajru.streamlit.app/
