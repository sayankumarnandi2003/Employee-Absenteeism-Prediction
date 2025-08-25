# train_model.py
# FINAL CORRECTED VERSION 
# This script now perfectly matches the preprocessing in your absenteeism_module.

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

print("Starting final model retraining process...")

# --- 1. Load The Preprocessed Training Data ---
try:
    df = pd.read_csv('Absenteeism_preprocessed (1).csv')
    print("Successfully loaded 'Absenteeism_preprocessed (1).csv'.")
except FileNotFoundError:
    print("\nERROR: 'Absenteeism_preprocessed (1).csv' not found.")
    print("Stopping script.")
    exit()

# --- 2. Create Target and Inputs ---
median_absenteeism = df['Absenteeism Time in Hours'].median()
targets = np.where(df['Absenteeism Time in Hours'] > median_absenteeism, 1, 0)
unscaled_inputs = df.drop(['Absenteeism Time in Hours'], axis=1)

# --- 3. CRITICAL FIX: Drop Columns to Match Prediction Module ---
# This new step ensures the training data has the exact same columns as the prediction data.
columns_to_drop = ['Day of the Week', 'Daily Work Load Average', 'Distance to Work']
unscaled_inputs = unscaled_inputs.drop(columns_to_drop, axis=1)

print("Inputs and targets created. Columns matched with prediction module.")

# --- 4. Scale the Data ---
print("Scaling features...")
scaler = StandardScaler()
inputs = scaler.fit_transform(unscaled_inputs)
print("Scaling complete.")

# --- 5. Train the Model ---
print("Splitting data and training the model...")
x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=42)
model = LogisticRegression(solver='liblinear')
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print(f"Model training complete. Accuracy: {accuracy:.4f}")

# --- 6. Save the Final Model and Scaler ---
print("Saving new 'model' and 'scaler' files...")
with open('model', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('scaler', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("\n✅ Success! Final 'model' and 'scaler' files have been created.")
print("You can now re-run your Jupyter Notebook cell.")

