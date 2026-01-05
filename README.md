# CO2 Emissions Prediction Model

## 📌 Overview
This project analyzes vehicle characteristics to predict CO₂ emissions using a multiple linear regression model. By examining features such as engine size, cylinder count, and fuel consumption, the model estimates the CO₂ emissions output of vehicles with high accuracy.

---

## 📊 Dataset
The model uses the `FuelConsumption.csv` dataset containing vehicle specifications and emission data. Key features include:
- **ENGINESIZE**: Engine displacement in liters
- **CYLINDERS**: Number of engine cylinders
- **FUELCONSUMPTION_COMB**: Combined fuel consumption (L/100km)
- **CO2EMISSIONS**: Target variable (g/km)

---

## 🚀 Features
- **Data Exploration**: Visual relationships between features and CO₂ emissions
- **Multiple Linear Regression**: Predictive modeling using scikit-learn
- **Model Evaluation**: Comprehensive metrics (R², MSE, RMSE, MAE)
- **Interactive Prediction**: User-input functionality for real-time predictions
- **Reproducible Splits**: Fixed random state for consistent results

---

## 🛠️ Installation & Usage

### Prerequisites
Ensure you have Python installed along with the following libraries:
```bash
pip install pandas matplotlib numpy scikit-learn
```

### Running the Model
1. Clone the repository:
```bash
git clone https://github.com/yourusername/co2-emissions-prediction.git
cd co2-emissions-prediction
```

2. Place `FuelConsumption.csv` in the project directory

3. Run the Jupyter notebook or Python script:
```bash
jupyter notebook CO2_Emissions_Prediction.ipynb
```

---

## 📈 Model Performance
The trained model achieves the following metrics:
- **R² Score**: 0.8760 (87.6% variance explained)
- **Mean Squared Error (MSE)**: 512.86
- **Root Mean Squared Error (RMSE)**: 22.65
- **Mean Absolute Error (MAE)**: 16.72

*Interpretation*: R² > 0.8 indicates a strong predictive model.

---

## 🔍 How It Works

### 1. Data Preparation
- Selects relevant features: `ENGINESIZE`, `CYLINDERS`, `FUELCONSUMPTION_COMB`
- Splits data into training (80%) and testing (20%) sets

### 2. Model Training
Uses `LinearRegression` from scikit-learn with equation:
```
CO₂ = 67.35 + 11.21×EngineSize + 7.16×Cylinders + 9.52×FuelConsumption
```

### 3. Prediction Example
For a vehicle with:
- Engine Size: 4.0L
- Cylinders: 4
- Fuel Consumption: 8.0 L/100km

**Predicted CO₂ Emissions**: 216.98 g/km

---

## 🖼️ Visualizations
The project includes scatter plots showing relationships between:
1. Fuel Consumption vs. CO₂ Emissions
2. Engine Size vs. CO₂ Emissions
3. Cylinder Count vs. CO₂ Emissions

---

## 📁 Project Structure
```
co2-emissions-prediction/
│
├── CO2_Emissions_Prediction.ipynb  # Main Jupyter notebook
├── FuelConsumption.csv              # Dataset
├── README.md                        # This file
└── requirements.txt                 # Dependencies (optional)
```

---

## ⚠️ Note
The warning about `numexpr` version is non-critical and doesn't affect model functionality. To resolve:
```bash
pip install --upgrade numexpr
```

---

## 🎯 Future Improvements
- Add more vehicle features (transmission type, fuel type)
- Experiment with other regression algorithms (Random Forest, Gradient Boosting)
- Implement cross-validation for more robust evaluation
- Deploy as a web application using Flask/Streamlit
