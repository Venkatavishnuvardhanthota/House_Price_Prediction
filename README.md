# 🏠 House Price Prediction

> Predicting residential property prices using regression models — built as Week 1 of my Data Science internship at **XYlofy AI**.

---

## 📌 Problem Statement

Real estate buyers and sellers often rely on guesswork or outdated comparisons to estimate a property's fair value. This project builds a regression model that predicts house prices based on structural property features — area, number of rooms, amenities, and location access — and identifies which features most strongly influence price.

---

## 📦 Dataset

| Attribute | Detail |
|---|---|
| Source | [Kaggle — Housing Prices Dataset (yasserh)](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset) |
| File | `Housing.csv` |
| Original Shape | 545 rows × 13 columns |
| Final Shape | 545 rows × 17 columns (after feature engineering) |
| Target Column | `price` (house price in Indian Rupees) |
| Price Range | ₹17.5 Lakhs — ₹1.33 Crores |

---

## 📁 Project Structure

```
HousePricePrediction/
│
├── data/
│   ├── Housing.csv                         # Original raw dataset (unmodified)
│   └── Housing_cleaned.csv                 # Cleaned + encoded + engineered dataset
│
├── notebooks/
│   └── analysis.ipynb                      # Main Jupyter Notebook (all 5 tasks)
│
├── charts/
│   ├── chart1_price_distribution.png       # Price histogram + log-transformed view
│   ├── chart2_correlation_heatmap.png      # Feature correlation heatmap
│   ├── chart3_actual_vs_predicted.png      # Actual vs Predicted — both models
│   ├── chart4_feature_importance.png       # Random Forest feature importance
│   └── chart5_price_vs_area_furnishing.png # Price vs Area by furnishing status
│
├── models/
│   ├── linear_regression_model.pkl         # Trained Linear Regression model
│   ├── random_forest_model.pkl             # Trained Random Forest model
│   └── scaler.pkl                          # Fitted StandardScaler
│
├── reports/
│   └── summary.docx                        # 1-page project summary report
│
├── requirements.txt                        # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/Venkatavishnuvardhanthota/HousePricePrediction.git
cd HousePricePrediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Launch the notebook**
```bash
jupyter notebook notebooks/analysis.ipynb
```

Run all cells from top to bottom. No additional configuration needed.

---

## 🔬 Methodology

### Task 1 — Data Loading & Exploration
- Loaded dataset using Pandas, inspected shape, dtypes, and value ranges
- Confirmed **zero missing values** and **zero duplicate rows**
- Identified 6 numeric and 7 categorical columns
- Noted strong right-skew in price distribution — most houses between ₹20–60L with a premium tail above ₹1Cr

### Task 2 — Data Cleaning & Encoding
- **6 binary columns** (yes/no) label-encoded to 0/1 using explicit `.map()` — avoids ambiguity of LabelEncoder
- **`furnishingstatus`** (3 categories) one-hot encoded with `drop_first=True` to eliminate multicollinearity
- pandas 2.0 `bool` dtype from `get_dummies` explicitly cast to `int64` to prevent silent sklearn coercion bugs
- All 12 original feature columns retained — no ID fields, timestamps, or irrelevant columns exist in this dataset

### Feature Engineering
Three new features created from existing columns — **no target variable used** (zero data leakage):

| Feature | Formula | Rationale |
|---|---|---|
| `total_rooms` | bedrooms + bathrooms | Captures overall property capacity |
| `area_per_room` | area ÷ total_rooms | Distinguishes spacious vs cramped layouts of equal size |
| `luxury_score` | AC + hot water + prefarea + guestroom + basement | Composite amenity signal (range: 0–5) |

### Task 3 — Model Building
- **Train/test split:** 80/20 → 436 training, 109 test rows (`random_state=42`)
- **StandardScaler:** fitted on training data only, then applied to test — prevents data leakage
- **GridSearchCV:** 108 hyperparameter combinations × 5-fold cross-validation = 540 models evaluated for Random Forest

---

## 📊 Results

| Metric | Linear Regression | Random Forest (Tuned) |
|---|---|---|
| MAE | ₹970,043 | ₹1,022,560 |
| RMSE | ₹1,324,507 | ₹1,401,497 |
| **R² Score** | **0.6583 ✅** | 0.5977 |

**Linear Regression outperformed the tuned Random Forest on every metric.**

Best RF hyperparameters found by GridSearchCV:
```
max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=100
```

---

## 💡 Key Findings

- **Area** is the single strongest predictor of house price — confirmed by both Random Forest feature importance and Linear Regression coefficients
- **Bathrooms** rank second — more predictive than bedrooms, signalling property tier
- **Furnished homes** command a price premium over unfurnished homes at identical area sizes — furnishing status is an independent price driver
- **Linear Regression beat Random Forest** even after exhaustive hyperparameter tuning — confirming that price relationships in this dataset are fundamentally linear
- Feature engineering improved both models (LR: +0.005 R², RF: +0.014 R²) — confirming domain-aware features add value even on small datasets
- The model is off by **₹9.7 lakhs on a typical prediction** — roughly 19.7% of the average house price

---

## 📈 Visualizations

| Chart | Description |
|---|---|
| Price Distribution | Raw histogram with mean/median lines + log-transformed view showing right-skew |
| Correlation Heatmap | Lower-triangle heatmap of all feature correlations with price |
| Actual vs Predicted | Both models side by side against a perfect-prediction reference line |
| Feature Importance | Random Forest importances — area dominates, followed by bathrooms and luxury_score |
| Price vs Area by Furnishing | Scatter plot confirming furnished premium is independent of area |

---

## 🏢 Business Recommendation

Real estate sellers should prioritise **furnishing, air conditioning, and hot water heating** before listing. These three factors drove the luxury score — the third most important price predictor after area and bathrooms. The trained Linear Regression model can serve as a **baseline pricing tool** to assist agents in setting initial asking prices, with a typical margin of error of ₹9.7 lakhs.

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x + Jupyter Notebook | Development environment |
| Pandas / NumPy | Data loading, cleaning, feature engineering |
| Scikit-learn | Models, GridSearchCV, evaluation metrics |
| Matplotlib / Seaborn | All 5 visualizations |
| Joblib | Model and scaler serialization |
| Git / GitHub | Version control — committed after each task |

---

## 📂 How to Use the Saved Model

```python
import joblib
import pandas as pd

# Load model and scaler
model  = joblib.load('models/linear_regression_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Prepare your input (must match training feature order)
new_data = pd.DataFrame([{
    'area': 5000, 'bedrooms': 3, 'bathrooms': 2,
    'stories': 2, 'mainroad': 1, 'guestroom': 0,
    'basement': 1, 'hotwaterheating': 0, 'airconditioning': 1,
    'parking': 1, 'prefarea': 1,
    'furnishingstatus_semi-furnished': 0,
    'furnishingstatus_unfurnished': 0,
    'total_rooms': 5, 'area_per_room': 1000.0, 'luxury_score': 3
}])

# Scale and predict
new_scaled  = scaler.transform(new_data)
prediction  = model.predict(new_scaled)
print(f"Predicted Price: ₹{prediction[0]:,.0f}")
```

---

## 👤 Author

**Venkata Vishnu Vardhan Thota**
AI & Data Science Intern — XYlofy AI
[GitHub](https://github.com/Venkatavishnuvardhanthota) · [LinkedIn](https://www.linkedin.com/in/venkata-vishnu-vardhan-thota/)

---

*Internship Week 1 Project — May 2026*