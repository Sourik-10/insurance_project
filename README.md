# Insurance Charge Predictor

A machine learning project that estimates annual US health insurance charges from personal and health details. It includes a data preparation and model training pipeline, a FastAPI prediction API, and a Streamlit web app.

> **Note:** Predictions are estimates from a model trained on a specific dataset. They are not quotes, medical advice, or a guarantee of actual insurance costs.

## Features

- Clean and explore the Medical Cost Personal Dataset.
- Engineer BMI categories, age bands, and a smoker/obesity interaction feature.
- Compare Linear Regression, Random Forest, and XGBoost regressors.
- Serve predictions through a FastAPI endpoint.
- Enter height and weight in the Streamlit app to calculate BMI automatically.

## Project structure

| File or folder | Purpose |
|---|---|
| `data/insurance.csv` | Source dataset |
| `01_clean_data.py` | Removes duplicate rows, normalizes categorical values, and saves cleaned data |
| `02_feature_engineering.py` | Builds derived features and saves the featured dataset |
| `03_eda.py` | Produces exploratory plots and data summaries |
| `04_modeling.py` | Trains and compares regression models, then saves the best model |
| `api.py` | FastAPI application with the `POST /predict` endpoint |
| `streamlit_app.py` | Web interface for entering details and viewing estimates |
| `requirements.txt` | Python dependencies |
| `models/` | Generated model files (not committed) |
| `outputs/` | Generated plots and model comparison (not committed) |

## Requirements

- Python 3.10 or newer
- pip

## Setup

Run these commands from the project root.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies and build the data and model artifacts:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python 01_clean_data.py
python 02_feature_engineering.py
python 03_eda.py
python 04_modeling.py
```

The model pipeline writes `models/best_model.pkl`; the analysis and comparison outputs are saved in `outputs/`. These files are generated locally and excluded from Git. Run the data preparation and modeling steps before starting the API.

## Run the application

Start the API in one terminal:

```bash
uvicorn api:app --reload
```

Start the Streamlit app in a second terminal:

```bash
streamlit run streamlit_app.py
```

Open the Streamlit URL printed in the terminal, usually `http://localhost:8501`. The API documentation is available at `http://127.0.0.1:8000/docs`.

The app asks for age, weight, and height, then calculates BMI using `weight_kg / height_m²`. Age is a separate model input; it is not part of the BMI formula. The app also collects sex, number of children, smoking status, and region. The model currently accepts ages 18–64 and BMI values greater than 10 and less than 60.

If the API runs at another address, change the FastAPI base URL in the app's sidebar or set `INSURANCE_API_URL` before launching Streamlit.

## API example

Send BMI directly when calling the API (height and weight are only collected by the Streamlit interface):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 45, "sex": "male", "bmi": 34, "children": 2, "smoker": "yes", "region": "southeast"}'
```

Example response:

```json
{
  "predicted_charges": 42404.48,
  "bmi_category": "obese",
  "smoker_obese_risk": true
}
```

The API returns HTTP 503 if the trained model file is missing. Run the setup and modeling steps above to create it.

## Modeling notes

The target is the numeric `charges` value, so this is a regression task. The training script uses an 80/20 train/test split with a fixed random seed and compares models using RMSE, MAE, and R². It selects the model with the lowest RMSE and saves it for API use. Results depend on the dataset and split; retrain the model after changing the data or feature pipeline.

The engineered BMI categories use cutoffs at 18.5, 25, and 30. `smoker_obese` flags records where smoking status is yes and BMI is at least 30.
