from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_thyroid_ann.keras"
SCALER_PATH = BASE_DIR / "scaler.pkl"
BLOOD_ENCODER_PATH = BASE_DIR / "blood_encoder.pkl"
FEATURES_PATH = BASE_DIR / "features.pkl"
FEATURE_WEIGHTS_PATH = BASE_DIR / "feature_weights.npy"


# ============================================================
# LOAD MODEL AND PREPROCESSING OBJECTS
# ============================================================

model = tf.keras.models.load_model(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

blood_encoder = joblib.load(BLOOD_ENCODER_PATH)

FEATURES = joblib.load(FEATURES_PATH)

feature_weights = np.load(FEATURE_WEIGHTS_PATH)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "Normal",
    1: "Hypothyroid",
    2: "Hyperthyroid",
}


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Thyroid Dysfunction Prediction API",
    description="ANN-based thyroid condition prediction API.",
    version="1.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# INPUT MODEL
# ============================================================

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


# ============================================================
# OUTPUT MODEL
# ============================================================

class PredictionResponse(BaseModel):
    predicted_class: int

    predicted_condition: str

    confidence: float

    probabilities: Dict[str, float]


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_patient(patient: PatientInput) -> np.ndarray:

    # The frontend sends height in centimeters.
    # The trained model receives height in meter-scale values.
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
            height_for_model,
        ]],
        columns=FEATURES,
    )

    # --------------------------------------------------------
    # Apply the same log transformation used during training
    # --------------------------------------------------------

    patient_data["TSH (mIU/L)"] = np.log1p(
        patient_data["TSH (mIU/L)"].astype(float)
    )

    patient_data["Total T4 (µg/dL)"] = np.log1p(
        patient_data["Total T4 (µg/dL)"].astype(float)
    )

    # --------------------------------------------------------
    # Encode blood group
    # --------------------------------------------------------

    encoded_blood_group = blood_encoder.transform(
        patient_data[["Blood Group"]]
    )

    patient_data["Blood Group"] = (
        encoded_blood_group[:, 0].astype(np.float32)
    )

    # --------------------------------------------------------
    # Convert data type
    # --------------------------------------------------------

    patient_data = patient_data.astype(np.float32)

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    patient_scaled = scaler.transform(patient_data)

    # --------------------------------------------------------
    # Apply saved feature weights
    # --------------------------------------------------------

    patient_scaled = patient_scaled * feature_weights

    patient_scaled = np.asarray(
        patient_scaled,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Validate ANN input shape
    # --------------------------------------------------------

    expected_shape = (1, len(FEATURES))

    if patient_scaled.shape != expected_shape:
        raise ValueError(
            f"Expected ANN input shape {expected_shape}, "
            f"received {patient_scaled.shape}"
        )

    return patient_scaled


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Thyroid Dysfunction Prediction API is running",
        "docs": "/docs",
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "feature_count": len(FEATURES),
        "classes": CLASS_NAMES,
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(patient: PatientInput):

    try:

        # ----------------------------------------------------
        # Preprocess patient
        # ----------------------------------------------------

        patient_scaled = preprocess_patient(patient)

        # ----------------------------------------------------
        # ANN prediction
        # ----------------------------------------------------

        probabilities = model.predict(
            patient_scaled,
            verbose=0,
        )[0]

        # ----------------------------------------------------
        # Predicted class
        # ----------------------------------------------------

        predicted_class = int(
            np.argmax(probabilities)
        )

        predicted_condition = CLASS_NAMES[
            predicted_class
        ]

        # ----------------------------------------------------
        # Prediction confidence
        # ----------------------------------------------------

        confidence = float(
            probabilities[predicted_class] * 100
        )

        # ----------------------------------------------------
        # Probability of each class
        # ----------------------------------------------------

        probabilities_dict = {
            "Normal": round(
                float(probabilities[0] * 100),
                4,
            ),

            "Hypothyroid": round(
                float(probabilities[1] * 100),
                4,
            ),

            "Hyperthyroid": round(
                float(probabilities[2] * 100),
                4,
            ),
        }

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return PredictionResponse(
            predicted_class=predicted_class,
            predicted_condition=predicted_condition,
            confidence=round(confidence, 4),
            probabilities=probabilities_dict,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc