# Insurance Charge Prediction

Regression project on the real Medical Cost Personal Dataset (1,337 records
after deduplication, US health insurance charges): cleaning → feature
engineering grounded in actual EDA findings → model comparison → FastAPI
serving layer.

**Deliberately different from the common tutorial version of this project:**
this is a regression problem (predicting actual $ charges), not a
Low/Medium/High classification — predicting the real number is strictly
harder and loses no information. Feature thresholds (the smoker×obese
interaction, BMI categories) come from what `03_eda.py` actually finds in
this data, not copied from a tutorial.

## How to run

```bash
pip install -r requirements.txt
python 01_clean_data.py
python 02_feature_engineering.py
python 03_eda.py
python 04_modeling.py

# Start the API
uvicorn api:app --reload
# Interactive docs at http://127.0.0.1:8000/docs
```

## Files

| File | What it does |
|---|---|
| `01_clean_data.py` | Loads raw CSV, drops the 1 known duplicate row, validates types/ranges |
| `02_feature_engineering.py` | WHO BMI categories, age bands, and a `smoker_obese` interaction flag |
| `03_eda.py` | 5 plots + data-computed interpretations (smoker effect, BMI×smoker interaction, age, children, region) |
| `04_modeling.py` | Linear Regression / Random Forest / XGBoost, compared on RMSE/MAE/R² |
| `api.py` | FastAPI service — `POST /predict` takes raw user input, computes derived features server-side, returns predicted charges |

## Key findings

- Smokers pay **3.8x** more on average ($32,050 vs $8,441) — by far the strongest driver
- The real interaction: BMI barely affects non-smokers' charges (spread of only $3,381 across BMI categories), but among smokers the spread is **$22,883** — obesity only matters once someone already smokes
- Age is only weakly correlated with charges (r=0.30) — much less predictive than smoking status
- Region and number of children are minor factors by comparison
- **Best model: Linear Regression** (RMSE $4,216, R² 90.3%) — the engineered
  `smoker_obese` feature apparently captures the key nonlinearity well enough
  that a linear model competes with (and here, slightly beats) tree-based models

## API usage

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 45, "sex": "male", "bmi": 34, "children": 2, "smoker": "yes", "region": "southeast"}'
```

Returns:
```json
{"predicted_charges": 42404.48, "bmi_category": "obese", "smoker_obese_risk": true}
```

## Streamlit frontend

Install the dependencies, then start the API and frontend in two terminals:

```bash
# Terminal 1 — from the project directory
uvicorn api:app --reload

# Terminal 2 — from the project directory
streamlit run streamlit_app.py
```

Open the Streamlit URL printed in the terminal (usually http://localhost:8501).
The app sends the form values to the FastAPI `/predict` endpoint. If the API
runs at a different address, set `INSURANCE_API_URL` or change the URL in the
app's sidebar.
