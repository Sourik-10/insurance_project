
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import Annotated, Literal
import joblib
import pandas as pd
import os

MODEL_PATH = "models/best_model.pkl"

app = FastAPI(title="Insurance Charge Predictor")

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


class UserInput(BaseModel):
    # The training data covers ages 18-64 and the age_group feature is built
    # for those bands. Keep API validation aligned with the model's domain.
    age: Annotated[int, Field(..., ge=18, le=64, description="Age in years (model training range)")]
    sex: Annotated[Literal["male", "female"], Field(..., description="Biological sex as recorded in the dataset")]
    bmi: Annotated[float, Field(..., gt=10, lt=60, description="Body mass index (kg/m^2)")]
    children: Annotated[int, Field(..., ge=0, le=10, description="Number of dependents covered")]
    smoker: Annotated[Literal["yes", "no"], Field(..., description="Smoking status")]
    region: Annotated[
        Literal["northeast", "northwest", "southeast", "southwest"],
        Field(..., description="US region of residence"),
    ]

    @computed_field
    @property
    def bmi_category(self) -> str:
        if self.bmi < 18.5:
            return "underweight"
        elif self.bmi < 25:
            return "normal"
        elif self.bmi < 30:
            return "overweight"
        return "obese"

    @computed_field
    @property
    def smoker_obese(self) -> int:
        return int(self.smoker == "yes" and self.bmi >= 30)

    @computed_field
    @property
    def age_group(self) -> str:
        if self.age <= 25:
            return "18-25"
        elif self.age <= 35:
            return "26-35"
        elif self.age <= 45:
            return "36-45"
        elif self.age <= 55:
            return "46-55"
        return "56-64"


@app.get("/")
def root():
    return {"message": "Insurance charge prediction API. See /docs for usage."}


@app.post("/predict")
def predict_charges(data: UserInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 01_clean_data.py, 02_feature_engineering.py, "
                   "and 04_modeling.py first to generate models/best_model.pkl."
        )

    input_df = pd.DataFrame([{
        "age": data.age,
        "bmi": data.bmi,
        "children": data.children,
        "sex": data.sex,
        "smoker": data.smoker,
        "region": data.region,
        "bmi_category": data.bmi_category,
        "age_group": data.age_group,
        "smoker_obese": data.smoker_obese,
    }])

    prediction = model.predict(input_df)[0]

    return JSONResponse(status_code=200, content={
        "predicted_charges": round(float(prediction), 2),
        "bmi_category": data.bmi_category,
        "smoker_obese_risk": bool(data.smoker_obese),
    })
