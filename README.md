🩺 Diabetes Detector (K-Nearest Neighbors Model)

A Python script that implements a simple Machine Learning (ML) model to predict the presence of diabetes (a binary classification task) based on various physiological and demographic factors.

📌 Features

✔ Machine Learning Approach: Uses the K-Nearest Neighbors (KNN) algorithm for classification.

✔ Data Handling: Utilizes pandas and scikit-learn for data processing and model training.

✔ Model Evaluation: Provides standard metrics like Accuracy, Precision, and Recall via a Classification Report.

✔ Predictive Function: Includes a dedicated diabetes_detector function to make predictions on new patient data.

✔ Confidence Scoring: Outputs the model's prediction along with the probability score (confidence).

🛠 Technologies Used

Python 3.x

pandas (Data manipulation)

numpy (Numerical operations)

scikit-learn (Machine learning library for model, data split, and metrics)

 How to Run the Project

Install Python: Ensure you have Python 3.x installed.

Install Dependencies: Install the required libraries via pip:

pip install pandas numpy scikit-learn


Save the Code: Save the Python script (the model logic) as diabetes_detector.py.

Execute: Open your terminal or command prompt in the directory where the file is saved.

Run the program: Use the following command:

python diabetes_detector.py


The console will output the model's training status, evaluation report, and the prediction for the example patient included in the script.

