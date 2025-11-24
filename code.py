import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_diabetes

print("Preparing the Diabetes Detection Model...")

data = {
    'Pregnancies': [6, 1, 8, 1, 0, 5, 3, 10, 2, 8],
    'Glucose': [148, 85, 183, 89, 137, 116, 78, 115, 197, 125],
    'BloodPressure': [72, 66, 64, 66, 40, 74, 50, 0, 70, 96],
    'SkinThickness': [35, 29, 0, 23, 35, 0, 32, 0, 45, 0],
    'Insulin': [0, 0, 0, 94, 168, 0, 88, 0, 543, 0],
    'BMI': [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 35.3, 30.5, 0.0],
    'DiabetesPedigreeFunction': [0.627, 0.351, 0.672, 0.167, 2.288, 0.201, 0.248, 0.134, 0.158, 0.232],
    'Age': [50, 31, 32, 21, 33, 30, 26, 29, 53, 54],
    'Outcome': [1, 0, 1, 0, 1, 0, 1, 0, 1, 1]  # 1 = Diabetic, 0 = Non-Diabetic
}
df = pd.DataFrame(data)



X = df.drop('Outcome', axis=1)
# Target (output): 'Outcome' column
y = df['Outcome']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = KNeighborsClassifier(n_neighbors=5)

print("\nTraining the K-Nearest Neighbors Model...")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# --- 6. Detector Function for New Patient ---
def diabetes_detector(new_patient_data, model):
    """
    Predicts diabetes for a new patient based on trained model.
    new_patient_data must be a DataFrame or Series with the same feature columns.
    """
    # Ensure the data is in a DataFrame format for the model
    if not isinstance(new_patient_data, pd.DataFrame):
        new_patient_data = pd.DataFrame([new_patient_data])
        
    # Make the prediction
    prediction = model.predict(new_patient_data)
    
    # Get the probability (confidence score)
    probability = model.predict_proba(new_patient_data)[0]
    
    # Interpret the result
    if prediction[0] == 1:
        result = "**DIABETES DETECTED**"
    else:
        result = "**No Diabetes Detected**"
        
    print("\n--- Patient Prediction ---")
    print(f"Prediction: {result}")
    print(f"Probability of No Diabetes (0): {probability[0]:.2f}")
    print(f"Probability of Diabetes (1): {probability[1]:.2f}")


new_patient = {
    'Pregnancies': 4,
    'Glucose': 160,
    'BloodPressure': 70,
    'SkinThickness': 35,
    'Insulin': 0,
    'BMI': 38.9,
    'DiabetesPedigreeFunction': 0.85,
    'Age': 40
}

diabetes_detector(new_patient, model)
