# 🏡 Real-Time Indian House Price Estimator

An end-to-end Machine Learning web application designed to estimate residential property valuations across Indian Metros and Tier cities with realistic market pricing (in ₹ Lakhs & ₹ Crores).

---

## 🌟 Key Features

- **Realistic Indian Real Estate Valuation**: Model calibrated for Indian market dynamics (Metro, Tier-1, Tier-2 cities).
- **Comprehensive Feature Analysis**:
  - **Location Tier**: Metro (Mumbai, Delhi NCR, Bengaluru), Tier-1 (Pune, Hyderabad, Jaipur), Tier-2 (Indore, Lucknow, Chandigarh)
  - **Carpet Area (sq. ft.)**: Usable living space
  - **BHK & Bathrooms**: Configurable from 1 to 5
  - **Property Age**: 0 to 30 years (with age depreciation dynamics)
  - **Furnishing Status**: Unfurnished, Semi-Furnished, Fully-Furnished
  - **Parking Availability**: Dedicated reserved parking slots
- **Advanced Machine Learning Pipeline**:
  - `ColumnTransformer` preprocessing with `StandardScaler` for numeric columns and `OneHotEncoder` for categorical features.
  - Regression benchmark comparison across Linear Regression, Random Forest, and Gradient Boosting.
  - Top-performing model serialized as `best_indian_house_model.pkl`.
- **Interactive Streamlit Web Dashboard**:
  - Instant real-time price estimation in both **₹ Lakhs** and **₹ Crores**.
  - Price per sq. ft. rate and estimated monthly home loan EMI calculations.
  - Interactive multi-tier price curves and cross-city comparison analytics.

---

## 📁 Project Structure

```
house-price-estimator/
│
├── data/
│   ├── houses.csv                     # Initial synthetic housing data
│   ├── generate_data.py               # Data generation script
│   └── indian_housing_dataset.csv     # 3,000 Indian housing records
│
├── charts/                            # EDA visualization outputs
├── app.py                             # Interactive Streamlit Web Application
├── train.py                           # ML Training and Model Evaluation Pipeline
├── eda.py                             # Exploratory Data Analysis script
├── best_indian_house_model.pkl        # Serialized Best Regression Pipeline
├── model.pkl                          # Compatible Model Pipeline artifact
├── requirements.txt                   # Python Dependencies
├── .gitignore                         # Git exclusion rules
└── README.md                          # Project Documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/mahakguptamg42-cell/house-price-estimator.git
cd house-price-estimator
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
```bash
python train.py
```

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Model Performance Highlights

| Model | $R^2$ Score | RMSE (₹ Lakhs) | MAE (₹ Lakhs) |
| :--- | :---: | :---: | :---: |
| **Gradient Boosting / XGBoost Regressor** | **0.8913** | **₹28.30 L** | **₹20.19 L** |
| Random Forest Regressor | 0.8827 | ₹29.39 L | ₹20.80 L |
| Linear Regression | 0.8275 | ₹35.65 L | ₹27.49 L |
