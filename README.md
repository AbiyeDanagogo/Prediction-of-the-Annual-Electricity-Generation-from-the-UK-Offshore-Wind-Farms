# Prediction of Annual Electricity Generation from UK Offshore Wind Farms

Investigating the meteorological factors that influence offshore wind turbine power output and developing machine learning models to forecast wind energy generation from SCADA time-series data.

---

## Overview

The UK is the world leader in offshore wind, targeting 40 GW of capacity by 2030 and 75 GW by 2050. Accurate power forecasting is critical to reducing operating costs, minimising wind curtailment, and maintaining grid reliability. Traditional physical methods using turbine power curves struggle with the intermittent and non-linear nature of wind, making data-driven approaches a more effective alternative.

This project uses meteorological mast and turbine SCADA data from the **7 MW Levenmouth Demonstration Turbine** in Fife, Scotland (data sourced from [ORE Catapult](https://pod.ore.catapult.org.uk/source/levenmouth-demonstration-turbine)) to train and evaluate multiple machine learning models for wind power forecasting.

---

## Dataset

Two datasets were used, both recorded at **10-minute intervals** from 2017 to 2021 via SCADA:

| Dataset | Source | Description |
|---|---|---|
| Met Mast | 11 sensors | Wind speed at 4 hub heights, temperature, air pressure, wind direction |
| Turbine | 574 sensors | Electrical output including active power generated |

---

## Methodology

### 1. Exploratory Analysis

- Descriptive statistics revealed significant data quality issues: extreme temperature outliers (minimum of -1.57 × 10¹⁶ °C), negative power values, and substantial missing data across features
- Wind rose analysis showed the dominant wind direction as south-westerly, with most speeds in the 6.2–12.3 m/s range
- Wind speed distribution was modelled using a Weibull fit (k = 1.92, A = 8.06 m/s)
- Autocorrelation analysis confirmed strong temporal dependence across all features, validating the use of lag-based and sequential approaches

### 2. Data Pre-processing

**Missing values** were handled using linear interpolation, which is appropriate given the high autocorrelation between adjacent time steps.

**Anomaly detection** was applied to the turbine power curve, where four distinct anomaly categories were identified:

- **Type 1**: No power output above cut-in wind speed (turbine downtime)
- **Type 2**: Curtailment — steady output below rated power
- **Type 3**: Scattered point anomalies from sensor faults or signal noise
- **Unidentified**: Power output recorded at zero wind speed

Approximately **44% of power readings** contained anomalies and required removal. A physics-based filter was first applied (removing readings below cut-in speed of 3 m/s), followed by a comparison of four outlier detection algorithms:

| Method | Notes |
|---|---|
| Local Outlier Factor | Best overall — identified the most Type 3 anomalies |
| Isolation Forest | Strong at identifying curtailment regions |
| Elliptical Envelope | Assumes Gaussian distribution |
| Gaussian Mixture Model | 3 clusters selected by BIC |

**Local Outlier Factor** was selected for the final pipeline.

### 3. Feature Engineering

The following features were engineered from the raw meteorological data:

| Feature | Description |
|---|---|
| U / V components | Vector decomposition of wind speed and direction to avoid angular discontinuity at 0°/360° |
| Wind shear (α) | Rate of change of wind speed with height using the power law exponent |
| Turbulence intensity | Standard deviation of wind speed divided by mean wind speed |
| Air density (ρ) | Derived from temperature and pressure: ρ = P / RT |

All features were normalised to [0, 1] to prevent scale bias.

### 4. Models

Four models were trained on an 80/20 train/test split with Wind Speed, Temperature, Pressure, U/V components, Wind Shear, Turbulence Intensity, and Air Density as inputs, and Power as the label:

| Model | Configuration |
|---|---|
| Random Forest | 200 decision trees, scikit-learn defaults |
| XGBoost | Up to 2,000 weak learners with early stopping |
| Neural Network (5-layer) | 1,000 epochs with early stopping |
| Neural Network (8-layer) | 1,000 epochs with early stopping |

---

## Results

| Model | Train RMSE | Train R² | Test RMSE | Test MAE | Test R² | Time (s) |
|---|---|---|---|---|---|---|
| Random Forest | 160.97 | 0.99 | 430.65 | 227.70 | **0.96** | 198 |
| XGBoost | 326.21 | 0.98 | 424.94 | 237.71 | **0.97** | 29.9 |
| 5-Layer NN | 527.95 | 0.95 | 524.82 | 269.17 | 0.95 | 788 |
| 8-Layer NN | 489.06 | 0.95 | 485.11 | 254.92 | 0.95 | 483 |

**XGBoost** achieved the best test R² (0.97) in the shortest training time. Random Forest had the lowest test MAE but showed greater overfitting. Both neural network architectures generalised well but underperformed the tree-based models on tabular data, consistent with findings in the literature.

### Feature Importance (XGBoost)

| Feature | Importance (%) |
|---|---|
| Wind Speed (110m) | 92.98 |
| Wind Shear | 1.91 |
| U Component | 1.04 |
| V Component | 0.95 |
| Turbulence Intensity | 0.92 |
| Air Density | 0.79 |
| Pressure | 0.75 |
| Temperature | 0.66 |

Wind speed at hub height is overwhelmingly the dominant predictor. While the engineered meteorological features contribute marginally, they have negligible independent importance.

---

## Key Findings

- Tree-based models (XGBoost, Random Forest) outperformed neural networks on this tabular dataset, achieving R² values of up to 0.97
- Wind speed is the primary driver of power output, accounting for ~93% of XGBoost feature importance
- Data quality is a significant challenge with SCADA systems — 44% of power readings required removal, which also prevented the use of sequential time series models that would otherwise be well-suited given the high autocorrelation in the data
- A time series approach (RNN, CNN, Transformer) could yield further improvements with a cleaner dataset or by taking sequential subsections with lower noise levels

---

## Tech Stack

- **Language**: Python
- **Data processing**: Pandas, NumPy
- **Visualisation**: Matplotlib
- **Machine learning**: scikit-learn, XGBoost
- **Deep learning**: TensorFlow / Keras

