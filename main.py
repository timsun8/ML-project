from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from catboost import CatBoostRegressor
from typing import Literal
import pandas as pd
import numpy as np
import json


app = FastAPI(title="PricePolice ML API")


# Config

with open("models/feature_columns.json", "r", encoding="utf-8") as f:
    FEATURE_COLUMNS = json.load(f)

CAT_FEATURES = ["form_sobs", "rayon", "street"]

SEGMENT_LIMITS = {
    "Квартира": {"min_area": 15, "max_area": 200, "max_price": 120_000_000},
    "Дом": {"min_area": 20, "max_area": 300, "max_price": 200_000_000},
    "Земельный участок": {"min_area": 10, "max_area": 300, "max_price": 200_000_000},
    "Помещение": {"min_area": 10, "max_area": 300, "max_price": 200_000_000},
}

FAIR_THRESHOLD = 10 


# Load models

def load_model(path):
    model = CatBoostRegressor()
    model.load_model(path)
    return model


apartment_model = load_model("models/apartment_model.cbm")
house_model = load_model("models/house_model.cbm")
land_model = load_model("models/land_model.cbm")
commercial_model = load_model("models/commercial_model.cbm")


# Input schema

class PredictionInput(BaseModel):
    property_type: Literal[
        "Квартира",
        "Дом",
        "Земельный участок",
        "Помещение"
    ] = Field(..., example="Квартира")

    form_sobs: str = Field(..., example="Индивидуальная")
    rayon: str = Field(..., example="р-н Бостандыкский")
    street: str = Field(..., example="ул. Абая")

    year: int = Field(..., ge=2022, le=2026, example=2024)
    quarter: int = Field(..., ge=1, le=4, example=2)
    plosh_ob: float = Field(..., gt=0, example=55.5)

    user_price: float | None = Field(
        default=None,
        gt=0,
        example=30000000,
        description="Optional user price for comparison"
    )


# Helpers

def select_model(property_type: str):
    if property_type == "Квартира":
        return apartment_model
    elif property_type == "Дом":
        return house_model
    elif property_type == "Земельный участок":
        return land_model
    elif property_type == "Помещение":
        return commercial_model
    else:
        raise ValueError("Unknown property_type")


def prepare_input(data: PredictionInput) -> pd.DataFrame:
    row = {
        "form_sobs": data.form_sobs,
        "rayon": data.rayon,
        "street": data.street,
        "year": data.year,
        "quarter": data.quarter,
        "log_plosh": np.log(data.plosh_ob)
    }

    X = pd.DataFrame([row])
    X = X[FEATURE_COLUMNS]

    for col in CAT_FEATURES:
        X[col] = X[col].astype(str)

    return X


def get_warnings(data: PredictionInput) -> list[str]:
    warnings = []
    limits = SEGMENT_LIMITS[data.property_type]

    if data.plosh_ob < limits["min_area"]:
        warnings.append(
            "Площадь ниже диапазона обучающих данных. Предсказание может быть менее точным."
        )

    if data.plosh_ob > limits["max_area"]:
        warnings.append(
            "Площадь выше диапазона обучающих данных. Предсказание может быть менее точным."
        )

    if data.user_price is not None and data.user_price > limits["max_price"]:
        warnings.append(
            "Указанная цена выше диапазона обучающих данных. Сравнение может быть менее точным."
        )

    if data.year > 2024:
        warnings.append(
            "Модель обучалась на данных 2022–2024 годов. Цены будущих периодов могут быть занижены."
        )

    return warnings


# Endpoints

@app.get("/")
def root():
    return {"message": "PricePolice ML API is running"}


@app.post("/predict")
def predict(data: PredictionInput):
    try:
        model = select_model(data.property_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    X = prepare_input(data)

    pred_log = model.predict(X)[0]
    predicted_price = float(np.exp(pred_log))

    warnings = get_warnings(data)

    response = {
        "property_type": data.property_type,
        "predicted_price": round(predicted_price, 2),
        "predicted_price_mln": round(predicted_price / 1_000_000, 2),
        "reliable": len(warnings) == 0
    }

    if warnings:
        response["warnings"] = warnings

    if data.user_price is not None:
        diff = data.user_price - predicted_price
        diff_percent = (diff / predicted_price) * 100

        if abs(diff_percent) <= FAIR_THRESHOLD:
            status = "fair"
            message = f"Ваша цена находится в пределах справедливого диапазона (±{FAIR_THRESHOLD}%)."
        elif diff_percent > FAIR_THRESHOLD:
            status = "overpriced"
            message = f"Ваша цена выше ожидаемой на {diff_percent:.2f}%."
        else:
            status = "underpriced"
            message = f"Ваша цена ниже ожидаемой на {abs(diff_percent):.2f}%."

        response.update({
            "user_price": round(data.user_price, 2),
            "difference": round(diff, 2),
            "difference_mln": round(diff / 1_000_000, 2),
            "difference_percent": round(diff_percent, 2),
            "status": status,
            "message": message
        })

    return response