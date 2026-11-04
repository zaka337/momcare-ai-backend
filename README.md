# MomCare AI backend

A FastAPI service that predicts maternal health risk as **low**, **mid** or **high**, with a confidence percentage for each class. It is the backend for MomCare AI, a maternal-care app idea: send a patient's vitals, lifestyle and background details, get a risk level back.

> **A prototype for demonstration.** It is not a medical device and its predictions are not medical advice.

## How it works

- **Model:** a Random Forest classifier (200 trees, balanced class weights) trained on a dataset of about 1,000 records with a stratified 80/20 train and test split.
- **17 input features:** vitals (age, systolic and diastolic blood pressure, blood sugar, body temperature, heart rate), lifestyle (sleep hours, physical activity, stress), nutrition (diet quality and adherence, hemoglobin, iron and folic supplements) and background (education, income, urban or rural).
- **Preprocessing:** ordered answers such as income level or diet adherence are mapped to numbers; the three risk labels are encoded for training and decoded back in the response.
- **API:** `POST /predict` validates every field with Pydantic and returns clear errors for bad input. The trained model and label encoder are saved with joblib and loaded once at start-up.

## Run it

```bash
pip install fastapi uvicorn pandas scikit-learn joblib
uvicorn app:app --reload
```

The interactive documentation is at http://127.0.0.1:8000/docs.

## API

`GET /` returns a short status message.

`POST /predict` takes all 17 fields:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "Age": 25, "SystolicBP": 130, "DiastolicBP": 80, "BS": 15, "BodyTemp": 98, "HeartRate": 86,
  "sleep_hours": 8.04, "phys_activity_score": 8.23, "stress_score": 3.99,
  "education_level": "Secondary", "income_category": "Medium", "urban_rural": "Urban",
  "diet_quality_score": 3.92, "hemoglobin_g_dL": 11.47,
  "iron_supplement": 1, "folic_supplement": 1, "diet_adherence": "Fair"
}'
```

Example response (the numbers depend on the trained model):

```json
{
  "status": "success",
  "prediction": "high risk",
  "confidence_scores_percentage": { "high risk": 89.5, "low risk": 1.5, "mid risk": 9.0 }
}
```

Bad input is rejected with a clear message: a missing or wrongly typed field gets a `422`, and an unknown value for a text field (for example `"income_category": "Rich"`) gets a `400` that lists the allowed values.

Allowed values for the text fields: `education_level` is `No formal`, `Primary`, `Secondary` or `Tertiary`; `income_category` is `Low`, `Medium` or `High`; `urban_rural` is `Rural` or `Urban`; `diet_adherence` is `Poor`, `Fair` or `Good`. `iron_supplement` and `folic_supplement` are `1` (yes) or `0` (no).

## Retrain the model

```bash
python train_model.py
```

This reads `Dataset.csv`, trains the model, prints a classification report, and writes `maternal_health_model.pkl` and `label_encoder.pkl` next to the script.

### How well does it work?

`Dataset.csv` has 1,014 records (406 low, 336 mid and 272 high risk). Retrained with the settings in `train_model.py` and tested on a random 20% hold-out (203 records):

| Class | Precision | Recall | Records |
|---|---|---|---|
| high risk | 0.91 | 0.89 | 55 |
| low risk | 0.70 | 0.79 | 81 |
| mid risk | 0.65 | 0.55 | 67 |

Overall accuracy is 0.74. Mid risk is the hardest class to tell apart from its neighbours. This is a small dataset and a single split, so treat the numbers as a demonstration of the pipeline, not as a clinically validated result.

If scikit-learn warns that the saved model was built with a different version, run `python train_model.py` to rebuild it for the version you have installed.

## Files

| File | What it is |
|---|---|
| `app.py` | The FastAPI service |
| `train_model.py` | Training script |
| `Dataset.csv` | The training data |
| `maternal_health_model.pkl`, `label_encoder.pkl` | The trained model and label encoder |

## Notes

- The model files are Python pickles. Only load pickle files you trust.
- CORS is open to every origin so that any frontend can call the API while developing. Restrict `allow_origins` in `app.py` before deploying it anywhere real.
- The model has been evaluated only on a hold-out split of its own training file. It has not been validated on any independent or clinical data.
