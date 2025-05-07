import os
import polars as pl
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib
from typing import Dict, List, Union
from . import dispatcher
from .eval_metrics import EvalMetrics
from .utils import create_fold_mapping

def train(
        data_df, 
        target, 
        model, 
        model_folder_path, 
        fold, 
        total_folds, 
        version, 
        eval_metric
        ):
    # Create fold mapping for cross-validation
    fold_mapping = create_fold_mapping(total_folds)
    
    # Load the dataset
    df = data_df
    
    # Split data into training and validation sets
    train_df = df.filter(pl.col('kfold').is_in(fold_mapping[fold]))
    valid_df = df.filter(pl.col('kfold') == fold)

    # Extract target variables
    ytrain = train_df[target]
    yvalid = valid_df[target]

    # Drop target and kfold columns from training and validation sets
    cols_to_drop = [target, 'kfold']
    colnames = train_df.drop(cols_to_drop).columns

    train_df = train_df.drop(cols_to_drop)
    valid_df = valid_df.drop(cols_to_drop)

    # Initialize and train the model
    t_model = dispatcher.MODELS[model]
    t_model.fit(train_df, ytrain)
    
    # Generate predictions and evaluate performance
    preds = dispatcher.get_proba_pred(t_model, valid_df)
    eval_cls = EvalMetrics()
    eval = eval_cls(eval_metric, yvalid, preds)
    print(eval)
    
    # Save model and column information
    model_path = f"{model_folder_path}{model}_{fold}_{version}.pkl"
    columns_path = f"{model_folder_path}{model}_{fold}_{version}_columns.pkl"
    
    joblib.dump(t_model, model_path)
    joblib.dump(colnames, columns_path)
