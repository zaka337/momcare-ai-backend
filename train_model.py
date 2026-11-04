import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split  # Fixed the typo here
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Automatically find the folder where train_model.py lives
base_dir = os.path.dirname(os.path.abspath(__file__))
file_name = os.path.join(base_dir, 'Dataset.csv')  # Points to Dataset.csv (the file's real name: matters on Linux and macOS)

# 1. Load the dataset
df = pd.read_csv(file_name)

print("--- Encoding Categorical Variables ---")
# Explicit ordinal mapping for categorical features
income_map = {'Low': 0, 'Medium': 1, 'High': 2}
diet_map = {'Poor': 0, 'Fair': 1, 'Good': 2}
edu_map = {'No formal': 0, 'Primary': 1, 'Secondary': 2, 'Tertiary': 3}
urban_map = {'Rural': 0, 'Urban': 1}

df['income_category'] = df['income_category'].map(income_map)
df['diet_adherence'] = df['diet_adherence'].map(diet_map)
df['education_level'] = df['education_level'].map(edu_map)
df['urban_rural'] = df['urban_rural'].map(urban_map)

# 2. Separate features (X) and target (y)
target_col = 'RiskLevel'
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode target labels ('high risk', 'mid risk', 'low risk')
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"Target classes detected: {label_encoder.classes_}")

# 3. Stratified Train-Test Split (80% training, 20% validation)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 4. Initialize and Train Model
print("\nTraining Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# 5. Evaluate the Model
y_pred = model.predict(X_test)
print("\n--- Phase 1: Model Performance Evaluation ---")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 6. Save Artifacts for Phase 2 (FastAPI)
joblib.dump(model, 'maternal_health_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
print("\nModel saved successfully as 'maternal_health_model.pkl'!")
print("Label Encoder saved successfully as 'label_encoder.pkl'!")