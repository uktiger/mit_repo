import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# Load the data using relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_data_path = os.path.join(BASE_DIR, '..', 'data', 'raw', 'train.csv')
test_data_path = os.path.join(BASE_DIR, '..', 'data', 'raw', 'test.csv')

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Separate the data
X_train = train_data.drop(columns=['loan_status'])
y_train = train_data['loan_status']

X_test = test_data.drop(columns=['loan_status'])
y_test = test_data['loan_status']

# One-Hot Encoding (OHE)
X_train_encoded = pd.get_dummies(X_train)
X_test_encoded = pd.get_dummies(X_test)

# Reindex test data to match the structure of train data
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# Feature scaling using MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)  # Note: Use `transform` not `fit_transform` for test set

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_encoded.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_encoded.columns)

# Concatenate input (X) and output (y) to create the final processed datasets
train_processed = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)
test_processed = pd.concat([X_test_scaled, y_test.reset_index(drop=True)], axis=1)

# Save the processed data locally
data_path = os.path.join(BASE_DIR, '..', 'data', 'processed')
os.makedirs(data_path, exist_ok=True)  # Avoid "directory already exists" error

train_processed.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
test_processed.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)

print(f"Processed data saved to {data_path}")
