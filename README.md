# 🌾 Crop Water Requirement Ledger

A Random Forest regression model + Streamlit app that estimates daily crop irrigation demand (mm/day) from weather and field conditions — built as a data science internship project.

## What it does

Given a crop type, soil type, and current field/weather readings (temperature, humidity, wind speed, solar radiation, rainfall, soil moisture, and days since sowing), the app predicts the crop's water requirement in mm/day, then converts that into liters/day and total liters for the field's acreage. It also cross-checks the prediction against every other crop under the same conditions, and shows feature importance and validation metrics for the underlying model.

## Repo contents

| File | Description |
|---|---|
| `app.py` | Streamlit web app ("Crop Water Ledger") — prediction UI, gauges, crop comparison, model diagnostics |
| `Crop_Water_Requirement_Prediction.ipynb` | Notebook covering data prep, feature engineering, model training (Linear Regression, Gradient Boosting, Random Forest) and evaluation |
| `crop_water_model.pkl` | Trained Random Forest model (best performer: R² = 0.761) |
| `crop_encoder.pkl` | `LabelEncoder` for crop type |
| `soil_encoder.pkl` | `LabelEncoder` for soil type |
| `scaler.pkl` | `StandardScaler` fitted on the training features |

All artifacts the app needs are included — clone and run.

## Model

Three models were trained on an 80/20 split of 1,500 synthetic field records:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Random Forest (deployed)** | 0.150 | 0.264 | **0.761** |
| Gradient Boosting | 0.182 | 0.281 | 0.729 |
| Linear Regression | 0.303 | 0.410 | 0.423 |

The target (`Water_Requirement_mm`) was engineered from a simplified reference evapotranspiration (ETo) estimate, scaled by crop coefficient and soil water-holding factor, and adjusted for rainfall, soil moisture, and growth stage.

This is a proof of concept on synthetic data, not agronomically validated field-sensor data — a natural next step is retraining on real regional weather/irrigation records.

## Running locally

```bash
git clone https://github.com/<your-username>/crop-water-ledger.git
cd crop-water-ledger
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Tech stack

Python · scikit-learn · pandas / numpy · Streamlit · Plotly
