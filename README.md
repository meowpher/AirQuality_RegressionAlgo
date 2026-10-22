# 🏭 Air Quality Regression — TYBCA Capstone Project

## BCA30030 | Unit 3: Supervised Regression

**Objective:** Predict the continuous Benzene concentration (`C6H6(GT)`) from the [UCI Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality) using six regression algorithms from the Unit 3 syllabus.

---

## 📂 Project Structure

```
AirQualityR/
├── AirQualityUCI.csv                 # Raw dataset (UCI Machine Learning Repository)
├── Air_Quality_Regression.ipynb      # Main Jupyter Notebook (41 code cells)
├── air_quality_regressor.pkl         # Exported champion model (joblib)
├── scaler.pkl                        # Exported StandardScaler (joblib)
├── predict.py                        # Standalone prediction script (optional)
└── README.md                         # This file
```

---

## 🧰 Prerequisites

Ensure the following Python libraries are installed:

| Library | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.24 | Numerical computation |
| `pandas` | ≥2.0 | Data manipulation |
| `matplotlib` | ≥3.7 | Plotting |
| `seaborn` | ≥0.12 | Statistical visualisation |
| `scikit-learn` | ≥1.3 | ML models & metrics |
| `joblib` | ≥1.3 | Model serialisation |

**Install all at once:**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

---

## 🚀 Execution Steps

### 1. Clone or download this project folder

Ensure `AirQualityUCI.csv` is in the **same directory** as the notebook.

### 2. Launch Jupyter Notebook

```bash
cd AirQualityR
jupyter notebook Air_Quality_Regression.ipynb
```

Or open the `.ipynb` file in **VS Code**, **JupyterLab**, or **Google Colab**.

### 3. Run All Cells

From the Jupyter menu bar:

```
Kernel → Restart & Run All
```

The notebook will execute all 41 cells sequentially:
- **Cells 1–4:** Import libraries and load the dataset.
- **Cells 5–10:** Data cleaning (drop empties, replace sentinels, impute NaNs).
- **Cells 11–16:** EDA (statistics, histogram, boxplots, heatmap, regplot).
- **Cells 17–20:** Leakage-free train/test split, IQR Winsorization, and scaling.
- **Cells 21–38:** Train and evaluate 6 regressors (3 cells each).
- **Cells 39–41:** Model comparison table, bar chart, and `.pkl` export.

### 4. Output Files

After execution, two `.pkl` files will be saved in the project folder:

| File | Contents |
|------|----------|
| `air_quality_regressor.pkl` | Best-performing regression model |
| `scaler.pkl` | Fitted `StandardScaler` (must be used to transform new inputs) |

---

## 📊 Models Trained (Unit 3 Syllabus)

| # | Algorithm | Scikit-learn Class |
|---|-----------|-------------------|
| 1 | Multiple Linear Regression | `LinearRegression` |
| 2 | Ridge Regression (L2) | `Ridge(alpha=1.0)` |
| 3 | Lasso Regression (L1) | `Lasso(alpha=0.1)` |
| 4 | Decision Tree Regressor | `DecisionTreeRegressor(max_depth=10)` |
| 5 | Random Forest Regressor | `RandomForestRegressor(n_estimators=200)` |
| 6 | Support Vector Regressor | `SVR(kernel='rbf', C=100)` |

### Evaluation Metrics

- **MAE** — Mean Absolute Error
- **MSE** — Mean Squared Error
- **RMSE** — Root Mean Squared Error
- **R²** — R-Squared (Coefficient of Determination)
- **Adjusted R²** — R² penalised for the number of features

---

## 🔮 How to Use for Predictions (`predict.py`)

After running the notebook, use the exported `.pkl` files to make predictions on **new sensor readings**. Save the following as `predict.py` or run it directly:

```python
"""
predict.py — Load the trained model and scaler to predict Benzene concentration.
Usage: python predict.py
"""
import numpy as np
import joblib

# Load the exported model and scaler
model = joblib.load('air_quality_regressor.pkl')
scaler = joblib.load('scaler.pkl')

# Define new sensor readings (mock input)
# Features: [PT08.S1(CO), PT08.S2(NMHC), NOx(GT), NO2(GT), T, RH]
new_data = np.array([[1200, 950, 180, 110, 15.5, 55.0]])

# Scale the input using the same scaler fitted during training
new_data_scaled = scaler.transform(new_data)

# Predict Benzene concentration
prediction = model.predict(new_data_scaled)

print(f'Input features:  {new_data[0]}')
print(f'Predicted C6H6(GT) (Benzene): {prediction[0]:.2f} µg/m³')
```

**Expected output:**

```
Input features:  [1200.  950.  180.  110.   15.5  55. ]
Predicted C6H6(GT) (Benzene): X.XX µg/m³
```

> **Note:** The exact predicted value depends on which model was selected as the champion during your notebook run.

---

## ⚠️ Important Notes

1. **Data Leakage Prevention:** The train/test split is performed *before* any outlier handling or scaling. IQR bounds and scaler statistics are computed exclusively on training data.

2. **Sentinel Values:** The raw dataset uses `-200` to indicate faulty sensor readings. These are replaced with `NaN` and imputed via forward-fill/backward-fill.

3. **Reproducibility:** All random operations use `random_state=42` for deterministic results.

---

*TYBCA — BCA30030 Machine Learning Capstone Project*
