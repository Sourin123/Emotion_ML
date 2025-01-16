import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

def clean_column_names(df):
    # Remove leading underscores and asterisks from column names
    return df.rename(columns=lambda x: x.strip('_*'))

# Function to process the data and extract features and labels
def process_data(data, is_training=True):
    # Clean the data
    data = clean_column_names(data)
    
    if is_training:
        # For training data, get labels from 'Class' column
        labels = data['Class']
        # Remove 'Class' column from features
        features = data.drop(['Class'], axis=1)
    else:
        # For test data, just return features
        features = data
        labels = None
    
    return features, labels

# Read the data
train_data = pd.read_csv('Acoustic_train.csv')
test_data = pd.read_csv('Acoustic_test_no_class.csv')

# Process training data
X_train, y_train = process_data(train_data, is_training=True)

# Process test data
X_test, _ = process_data(test_data, is_training=False)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the SVM classifier
svm_classifier = SVC(kernel='rbf', random_state=42)
svm_classifier.fit(X_train_scaled, y_train)

# Make predictions on test data
y_pred = svm_classifier.predict(X_test_scaled)

# Save predictions to a CSV file
predictions_df = pd.DataFrame({
    'Predicted_Class': y_pred
})
predictions_df.to_csv('predictions.csv', index=False)

print("\nPredictions have been saved to 'predictions.csv'")

# Save the model and scaler
from joblib import dump
dump(svm_classifier, 'emotion_svm_model.joblib')
dump(scaler, 'emotion_scaler.joblib')

""" Function to predict emotion for new data"""
def predict_emotion(new_data):
    # Clean and process new data
    new_data = clean_column_names(new_data)
    # Scale the new data using the same scaler
    new_data_scaled = scaler.transform(new_data)
    # Make prediction
    prediction = svm_classifier.predict(new_data_scaled)
    return prediction