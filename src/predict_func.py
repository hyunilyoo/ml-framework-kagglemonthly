import os
import polars as pl
import pandas as pd
import joblib
import numpy as np
from . import dispatcher

TEST_DATA = os.environ.get('TEST_DATA')
SUBMIT = os.environ.get('SUBMIT')
NUM_FOLD = int(os.environ.get('NUM_FOLD'))
MODEL = os.environ.get('MODEL')
MODEL_PATH = os.environ.get('MODEL_PATH')
VERSION = os.environ.get('VERSION')

def clf_predict(test_data_path: str, model_type: str, model_name: str, model_path: str, total_folds: int) -> pd.DataFrame:
    """
    Generate predictions by ensembling multiple fold models.
    
    Args:
        test_data_path: Path to the test CSV file
        model_type: Type of model (e.g., 'lgbm', 'xgb')
        model_name: Name/version of the model
        model_path: Directory containing saved model files
    
    Returns:
        DataFrame with id and target prediction columns
    """
    # Load test data
    df = pl.read_csv(test_data_path)
    test_idx = np.array(df['id'])
    df = df.drop('id')
    
    # Initialize predictions array
    predictions = None
    
    # Generate and ensemble predictions from each fold
    for fold in range(NUM_FOLD):
        # Load model for current fold
        model_filename = f"{model_type}_{fold}_{model_name}.pkl"
        model_file_path = os.path.join(model_path, model_filename)
        clf = joblib.load(model_file_path)
        
        # Get predictions for current fold
        fold_preds = dispatcher.get_probability_predictions(clf, df)
        
        # Add to ensemble
        if predictions is None:
            predictions = fold_preds
        else:
            predictions += fold_preds
    
    # Average predictions across all folds
    predictions /= NUM_FOLD
    
    # Create submission dataframe
    submission = pd.DataFrame({
        "id": test_idx, 
        "target": predictions
    })
    
    return submission


if __name__ == "__main__":
    # Generate predictions and save submission file
    submission = clf_predict(
        test_data_path=TEST_DATA,
        model_type=MODEL,
        model_name=VERSION,
        model_path=MODEL_PATH,
        total_folds=NUM_FOLD
    )
    
    submission.to_csv(SUBMIT, index=False)
