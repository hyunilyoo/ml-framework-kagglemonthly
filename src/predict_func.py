import os
import polars as pl
import pandas as pd
import joblib
import numpy as np
from . import dispatcher

def predict(test_data, model, model_name, model_path, total_folds: int):
    # Load test data
    df = test_data 
    test_idx = df['id']
    df = df.drop(['id', 'Sex'], axis=1)
    
    # Initialize predictions array
    predictions = None
    
    # Generate and ensemble predictions from each fold
    for fold in range(total_folds):
        # Load model for current fold
        model_filename = f"{model}_{fold}_{model_name}.pkl"
        model_file_path = os.path.join(model_path, model_filename)
        preds = joblib.load(model_file_path)
        
        # Get predictions for current fold
        fold_preds = dispatcher.get_proba_pred(preds, df)
        
        # Add to ensemble
        if predictions is None:
            predictions = fold_preds
        else:
            predictions += fold_preds
    
    # Average predictions across all folds
    predictions /= total_folds
    
    # Create submission dataframe
    submission = pd.DataFrame({
        "id": test_idx, 
        "target": predictions
    })
    
    return submission