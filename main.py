from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_thyroid_ann.keras"
SCALER_PATH = BASE_DIR / "scaler.pkl"
BLOOD_ENCODER_PATH = BASE_DIR / "blood_encoder.pkl"
FEATURES_PATH = BASE_DIR / "features.pkl"
FEATURE_WEIGHTS_PATH = BASE_DIR / "feature_weights.npy"
SHAP_BACKGROUND_PATH = BASE_DIR / "shap_background.npy"


model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
blood_encoder = joblib.load(BLOOD_ENCODER_PATH)
FEATURES = joblib.load(FEATURES_PATH)
feature_weights = np.load(FEATURE_WEIGHTS_PATH)
shap_background = np.load(SHAP_BACKGROUND_PATH)

shap_explainer = shap.GradientExplainer(
    model,
    shap_background
)

CLASS_NAMES = {
    0: "Normal",
    1: "Hypothyroid",
    2: "Hyperthyroid"
}


app = FastAPI(
    title="Thyroid Dysfunction Prediction API",
    description="ANN-based thyroid condition prediction with patient-specific SHAP explanation.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PatientInput(BaseModel):
    age: int = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    weight: float = Field(..., ge=20, le=200)
    height_cm: float = Field(..., ge=100, le=220)

    systolic_bp: float = Field(..., ge=60, le=250)
    diastolic_bp: float = Field(..., ge=40, le=150)

    blood_group: str
    blood_glucose: float = Field(..., ge=40, le=500)
    bmi: float = Field(..., ge=10, le=60)

    residential_location: int = Field(..., ge=0, le=1)
    irritability: int = Field(..., ge=0, le=1)
    fatigue: int = Field(..., ge=0, le=1)
    body_pain: int = Field(..., ge=0, le=1)

    tsh: float = Field(..., ge=0, le=200)
    total_t4: float = Field(..., ge=0, le=60)
    total_t3_status: int = Field(..., ge=0, le=1)

    pregnancy_status: int = Field(..., ge=0, le=1)
    other_disease: int = Field(..., ge=0, le=1)
    appetite_status: int = Field(..., ge=0, le=1)


class ShapItem(BaseModel):
    rank: int
    parameter: str
    shap_contribution: float
    absolute_shap_contribution: float
    direction: str


class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_condition: str
    confidence: float
    probabilities: Dict[str, float]
    shap_top_10: List[ShapItem]
    shap_all: List[ShapItem]


def preprocess_patient(patient: PatientInput) -> np.ndarray:
    height_for_model = patient.height_cm / 100.0

    patient_data = pd.DataFrame(
        [[
            patient.age,
            patient.sex,
            patient.weight,
            patient.systolic_bp,
            patient.diastolic_bp,
            patient.blood_group,
            patient.blood_glucose,
            patient.bmi,
            patient.residential_location,
            patient.irritability,
            patient.fatigue,
            patient.body_pain,
            patient.tsh,
            patient.total_t4,
            patient.total_t3_status,
            patient.pregnancy_status,
            patient.other_disease,
            patient.appetite_status,
            height_for_model
        ]],
        columns=FEATURES
    )

    patient_data["TSH (mIU/L)"] = np.log1p(
        patient_data["TSH (mIU/L)"].astype(float)
    )

    patient_data["Total T4 (µg/dL)"] = np.log1p(
        patient_data["Total T4 (µg/dL)"].astype(float)
    )

    encoded_blood_group = blood_encoder.transform(
        patient_data[["Blood Group"]]
    )

    patient_data["Blood Group"] = encoded_blood_group[:, 0].astype(np.float32)
    patient_data = patient_data.astype(np.float32)

    patient_scaled = scaler.transform(patient_data)
    patient_scaled = patient_scaled * feature_weights
    patient_scaled = np.asarray(patient_scaled, dtype=np.float32)

    expected_shape = (1, len(FEATURES))

    if patient_scaled.shape != expected_shape:
        raise ValueError(
            f"Expected ANN input shape {expected_shape}, "
            f"received {patient_scaled.shape}"
        )

    return patient_scaled


def normalize_shap_values(shap_values_raw) -> np.ndarray:
    if isinstance(shap_values_raw, list):
        shap_values_array = np.stack(
            [np.asarray(v) for v in shap_values_raw],
            axis=-1
        )
    else:
        shap_values_array = np.asarray(shap_values_raw)

    if shap_values_array.ndim == 2:
        shap_values_array = shap_values_array[:, :, np.newaxis]

    elif shap_values_array.ndim == 3:
        if shap_values_array.shape[1] == len(FEATURES):
            pass
        elif shap_values_array.shape[2] == len(FEATURES):
            shap_values_array = np.transpose(
                shap_values_array,
                (0, 2, 1)
            )
        else:
            raise ValueError(
                f"Unexpected SHAP output shape: "
                f"{shap_values_array.shape}"
            )
    else:
        raise ValueError(
            f"Unexpected SHAP dimensions: "
            f"{shap_values_array.shape}"
        )

    return shap_values_array


@app.get("/")
def root():
    return {
        "message": "Thyroid Dysfunction Prediction API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "feature_count": len(FEATURES),
        "classes": CLASS_NAMES
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    try:
        patient_scaled = preprocess_patient(patient)

        probabilities = model.predict(
            patient_scaled,
            verbose=0
        )[0]

        predicted_class = int(np.argmax(probabilities))
        predicted_condition = CLASS_NAMES[predicted_class]
        confidence = float(
            probabilities[predicted_class] * 100
        )

        shap_values_raw = shap_explainer.shap_values(
            patient_scaled
        )

        shap_values_array = normalize_shap_values(
            shap_values_raw
        )

        patient_shap_values = shap_values_array[
            0,
            :,
            predicted_class
        ]

        shap_df = pd.DataFrame({
            "parameter": FEATURES,
            "shap_contribution": patient_shap_values,
            "absolute_shap_contribution": np.abs(
                patient_shap_values
            )
        })

        shap_df["direction"] = shap_df[
            "shap_contribution"
        ].apply(
            lambda value: (
                f"Toward {predicted_condition}"
                if value > 0
                else (
                    f"Away from {predicted_condition}"
                    if value < 0
                    else "Neutral"
                )
            )
        )

        shap_df = shap_df.sort_values(
            by="absolute_shap_contribution",
            ascending=False
        ).reset_index(drop=True)

        shap_df.insert(
            0,
            "rank",
            range(1, len(shap_df) + 1)
        )

        shap_all = [
            ShapItem(
                rank=int(row["rank"]),
                parameter=str(row["parameter"]),
                shap_contribution=float(
                    row["shap_contribution"]
                ),
                absolute_shap_contribution=float(
                    row["absolute_shap_contribution"]
                ),
                direction=str(row["direction"])
            )
            for _, row in shap_df.iterrows()
        ]

        probabilities_dict = {
            "Normal": round(
                float(probabilities[0] * 100),
                4
            ),
            "Hypothyroid": round(
                float(probabilities[1] * 100),
                4
            ),
            "Hyperthyroid": round(
                float(probabilities[2] * 100),
                4
            )
        }

        return PredictionResponse(
            predicted_class=predicted_class,
            predicted_condition=predicted_condition,
            confidence=round(confidence, 4),
            probabilities=probabilities_dict,
            shap_top_10=shap_all[:10],
            shap_all=shap_all
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        ) from exc
