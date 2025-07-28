# Biomedical Data Science Platform Prototype

## Features

- Data upload (CSV/Excel), preview, and missing value/statistical summary
- Dynamic variable selection (Y and X)
- Preprocessing: StandardScaler, Min-Max, RobustScaler (with side-by-side comparison)
- Exploratory Analysis: Correlation heatmap, PCA
- ML: Linear/Logistic Regression, Random Forest, XGBoost; automatic task detection
- Model evaluation: R²/RMSE for regression, Accuracy/ROC-AUC/ROC curve for classification
- Comparative tables for all normalization & ML model combinations

## File Structure

- `app.py` — Main Streamlit app
- `modules/` — Core logic for data, preprocessing, EDA, modeling
- `requirements.txt` — Dependencies

## How to Run

1. **Clone or download this folder.**
2. *(Recommended)* Create a fresh Python environment.
3. **Install dependencies:**
