from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

app = FastAPI(title="MomCare AI Maternal Risk Prediction API")

# Enable CORS so your frontend can communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the saved AI artifacts
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'maternal_health_model.pkl')
encoder_path = os.path.join(base_dir, 'label_encoder.pkl')

try:
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    print("AI Model and Encoder loaded successfully!")
except Exception as e:
    print(f"Error loading model files: {e}")

# Explicitly typed fields to satisfy Python 3.14 + Pydantic requirements
class PatientData(BaseModel):
    Age: int = Field(..., description="Age of the patient")
    SystolicBP: int = Field(..., description="Systolic Blood Pressure")
    DiastolicBP: int = Field(..., description="Diastolic Blood Pressure")
    BS: float = Field(..., description="Blood Sugar levels")
    BodyTemp: float = Field(..., description="Body Temperature")
    HeartRate: int = Field(..., description="Heart Rate")
    sleep_hours: float = Field(..., description="Average daily sleep hours")
    phys_activity_score: float = Field(..., description="Physical activity score")
    stress_score: float = Field(..., description="Stress score level")
    education_level: str = Field(..., description="'No formal', 'Primary', 'Secondary', 'Tertiary'")
    income_category: str = Field(..., description="'Low', 'Medium', 'High'")
    urban_rural: str = Field(..., description="'Rural' or 'Urban'")
    diet_quality_score: float = Field(..., description="Diet quality index")
    hemoglobin_g_dL: float = Field(..., description="Hemoglobin levels")
    iron_supplement: int = Field(..., description="Iron supplement intake: 1 for Yes, 0 for No")
    folic_supplement: int = Field(..., description="Folic supplement intake: 1 for Yes, 0 for No")
    diet_adherence: str = Field(..., description="'Poor', 'Fair', 'Good'")

@app.get("/")
def home():
    return {"message": "MomCare AI Backend API is running successfully!"}

@app.post("/predict")
def predict_risk(patient: PatientData):
    try:
        # Maps incoming text strings to numerical values
        income_map = {'Low': 0, 'Medium': 1, 'High': 2}
        diet_map = {'Poor': 0, 'Fair': 1, 'Good': 2}
        edu_map = {'No formal': 0, 'Primary': 1, 'Secondary': 2, 'Tertiary': 3}
        urban_map = {'Rural': 0, 'Urban': 1}

        # Check for matching keys
        if patient.income_category not in income_map:
            raise HTTPException(status_code=400, detail="Invalid income_category. Use 'Low', 'Medium', or 'High'.")
        if patient.diet_adherence not in diet_map:
            raise HTTPException(status_code=400, detail="Invalid diet_adherence. Use 'Poor', 'Fair', or 'Good'.")
        if patient.education_level not in edu_map:
            raise HTTPException(status_code=400, detail="Invalid education_level. Use 'No formal', 'Primary', 'Secondary', 'Tertiary'.")
        if patient.urban_rural not in urban_map:
            raise HTTPException(status_code=400, detail="Invalid urban_rural. Use 'Rural' or 'Urban'.")

        # Compile values into expected dimensional array
        features = np.array([[
            patient.Age,
            patient.SystolicBP,
            patient.DiastolicBP,
            patient.BS,
            patient.BodyTemp,
            patient.HeartRate,
            patient.sleep_hours,
            patient.phys_activity_score,
            patient.stress_score,
            edu_map[patient.education_level],
            income_map[patient.income_category],
            urban_map[patient.urban_rural],
            patient.diet_quality_score,
            patient.hemoglobin_g_dL,
            patient.iron_supplement,
            patient.folic_supplement,
            diet_map[patient.diet_adherence]
        ]])

        # Predict numerical classification
        numeric_prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        # Convert back to readable string labels
        readable_prediction = label_encoder.inverse_transform([numeric_prediction])[0]

        class_probabilities = {
            label_encoder.classes_[i]: round(float(probabilities[i]) * 100, 2)
            for i in range(len(label_encoder.classes_))
        }

        return {
            "status": "success",
            "prediction": readable_prediction,
            "confidence_scores_percentage": class_probabilities
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))