import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1️⃣ Configure logging
logging.basicConfig(
    filename='data_ingestion.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data(file_path):
    """
    Load CSV data into a DataFrame.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        logging.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {str(e)}")
        raise

def split_data(df, target_column, drop_columns=None, test_size=0.25, random_state=0):
    """
    Split the DataFrame into train and test sets.
    Args:
        df (pd.DataFrame): The DataFrame to split.
        target_column (str): The name of the target column.
        drop_columns (list, optional): Columns to drop from input features.
        test_size (float, optional): The proportion of the test data.
        random_state (int, optional): Random state for reproducibility.
    Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame: x_train, x_test, y_train, y_test
    """
    try:
        logging.info("Splitting data into train and test sets.")
        x = df.drop(columns=drop_columns, axis=1)
        y = df[target_column]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        logging.info("Data split successfully.")
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error splitting data: {str(e)}")
        raise

def save_data(train_data, test_data, output_dir='data/raw'):
    """
    Save the train and test datasets as CSV files.
    Args:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.
        output_dir (str): Directory to save the files.
    """
    try:
        logging.info(f"Saving data to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        train_path = os.path.join(output_dir, 'train.csv')
        test_path = os.path.join(output_dir, 'test.csv')
        
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        logging.info(f"Data saved successfully at {output_dir}")
    except Exception as e:
        logging.error(f"Error saving data to {output_dir}: {str(e)}")
        raise

def main():
    """
    Main function to execute the data ingestion pipeline.
    """
    try:
        logging.info("Starting data ingestion process.")
        
        # Path to the dataset
        file_path = r'data/external/loan_approval_dataset.csv'  # Change to relative path for better reproducibility
        
        # 1. Load the data
        df = load_data(file_path)
        
        # 2. Split the data into train and test sets
        x_train, x_test, y_train, y_test = split_data(
            df, 
            target_column='loan_status', 
            drop_columns=['loan_id', 'loan_status']
        )
        
        # 3. Combine the input and target for train and test data
        train_data = pd.concat([x_train, y_train], axis=1)
        test_data = pd.concat([x_test, y_test], axis=1)
        
        # 4. Save the train and test datasets locally
        save_data(train_data, test_data)
        
        logging.info("Data ingestion pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Data ingestion pipeline failed: {str(e)}")

if __name__ == "__main__":
    main()
