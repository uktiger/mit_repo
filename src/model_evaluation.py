import os
import json
import logging
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score


def setup_logger():
    """Set up the logging configuration."""
    logging.basicConfig(
        filename='model_evaluation.log',
        filemode='a',  # append logs to the file
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_model(model_path: str):
    """Load the trained model from a pickle file."""
    try:
        logging.info(f"Loading model from {model_path}...")
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logging.info("✅ Model loaded successfully.")
        return model
    except FileNotFoundError:
        logging.error(f"❌ Model file not found at {model_path}. Please check the path.")
        raise
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred while loading the model: {e}")
        raise


def load_test_data(test_data_path: str) -> pd.DataFrame:
    """Load the test dataset from a CSV file."""
    try:
        logging.info(f"Loading test data from {test_data_path}...")
        test_data = pd.read_csv(test_data_path)
        logging.info(f"✅ Test data loaded successfully. Shape: {test_data.shape}")
        return test_data
    except FileNotFoundError:
        logging.error(f"❌ Test data file not found at {test_data_path}. Please check the path.")
        raise
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred while loading test data: {e}")
        raise


def evaluate_model(model, x_test, y_test):
    """Evaluate the model on the test data and calculate accuracy, precision, and recall."""
    try:
        logging.info("🤖 Making predictions on test data...")
        y_pred = model.predict(x_test)
        
        # Clean target labels if they contain whitespace
        y_test = [label.strip() for label in y_test] if isinstance(y_test[0], str) else y_test
        y_pred = [label.strip() for label in y_pred] if isinstance(y_pred[0], str) else y_pred
        
        logging.info("🧮 Calculating evaluation metrics...")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='Approved')
        recall = recall_score(y_test, y_pred, pos_label='Approved')
        
        metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall}
        logging.info(f"✅ Metrics calculated successfully: {metrics}")
        return metrics
    except Exception as e:
        logging.error(f"❌ An error occurred during model evaluation: {e}")
        raise


def save_metrics(metrics: dict, output_path: str):
    """Save the evaluation metrics to a JSON file."""
    try:
        logging.info(f"💾 Saving metrics to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=5)
        logging.info("✅ Metrics saved successfully.")
    except Exception as e:
        logging.error(f"❌ Failed to save metrics to {output_path}: {e}")
        raise


def main():
    """Main function to execute all the steps of model evaluation."""
    setup_logger()
    
    # Define file paths
    model_path = 'model.pkl'
    test_data_path = os.path.join('data', 'processed', 'test_processed.csv')
    metrics_output_path = 'reports/metrics.json'

    try:
        # Step 1: Load Model
        model = load_model(model_path)

        # Step 2: Load Test Data
        test_data = load_test_data(test_data_path)

        # Step 3: Split Test Data into X and Y
        logging.info("📊 Splitting test data into X (features) and Y (target)...")
        x_test = test_data.iloc[:, 0:-1].values
        y_test = test_data.iloc[:, -1].values

        # Step 4: Evaluate Model
        metrics = evaluate_model(model, x_test, y_test)

        # Step 5: Save Metrics to JSON
        save_metrics(metrics, metrics_output_path)

    except Exception as e:
        logging.error(f"❌ The process failed: {e}")
        print(f"❌ The process failed. Check model_evaluation.log for details.")
        

if __name__ == "__main__":
    main()