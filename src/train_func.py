import os
import polars as pl
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib
from typing import Dict, List, Union
from . import dispatcher
from .eval_metrics import EvalMetrics

# Create a mapping of each fold index to all other fold indices
def create_fold_mapping(total_folds: int) -> Dict[int, List[int]]:
    fold_mapping = {}

    for i in range(total_folds):
        remaining_folds = list(range(total_folds))
        remaining_folds.pop(i)  # Remove current fold
        fold_mapping[i] = remaining_folds
    return fold_mapping

def train(
    data_path: str, 
    target: str, 
    model: str, 
    model_folder_path: str, 
    fold: int, 
    total_folds: int,
    version: str,
    eval_metric: str
) -> None:
    """
    Train a clf/reg model using cross-validation
    
    Args:
        data_path: Path to the CSV file containing the dataset
        target: Name of the target column
        model: Model name (must be a key in dispatcher.MODELS)
        model_folder_path: Directory path where models will be saved
        fold: Current fold index for validation
        total_folds: Total number of folds for cross-validation
        version: Additional identifier for the saved model
        
    Returns:
        None: Prints eval metric score and saves model to model folder
    """
    # Create fold mapping for cross-validation
    fold_mapping = create_fold_mapping(total_folds)
    
    # Load the dataset
    df = pl.read_csv(data_path)
    
    # Split data into training and validation sets
    train_df = df.filter(pl.col('kfold').is_in(fold_mapping[fold]))
    valid_df = df.filter(pl.col('kfold') == fold)

    # Extract target variables
    ytrain = train_df[target]
    yvalid = valid_df[target]

    # Drop target and kfold columns from training and validation sets
    cols_to_drop = [target, 'kfold']
    train_df = train_df.drop(cols_to_drop)
    valid_df = valid_df.drop(cols_to_drop)

    # Initialize and train the model
    model = dispatcher.MODELS[model]
    model.fit(train_df, ytrain)
    
    # Generate predictions and evaluate performance
    preds = dispatcher.get_proba_pred(model, valid_df)
    eval_cls = EvalMetrics()
    eval = eval_cls(eval_metric, yvalid, preds)
    print(eval)
    
    # Save model and column information
    model_path = f"{model_folder_path}{model}_{fold}_{version}.pkl"
    columns_path = f"{model_folder_path}{model}_{fold}_{version}_columns.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(train_df.columns, columns_path)
