import pandas as pd
import os
from sklearn import model_selection
import time
import sys

def insert_kfold(dataFrame, target, n_split, seed = 42, verbose = True) -> pd.DataFrame:
    # TODO: pd to pl
    """
    Insert k-fold column into the dataset
    """
    start_time = time.time()
    # Load data
    df = dataFrame
    
    # Add k-fold column and shuffle
    if verbose:
        print(f"Data loaded and preprocessed in {time.time() - start_time:.2f} seconds")
        print(f"Dataset shape: {df.shape}")
        print("Sample data:")
        print(df.head())
        
    # Stratified k-fold split
    # Handle regression and classification
    if df[target].dtype == 'float64' or df[target].dtype == 'float32':
        kf = model_selection.KFold(n_splits=n_split, shuffle=False)
        kf.get_n_splits(df.loc[:, df.columns != target])
    else:
        kf = model_selection.StratifiedKFold(n_splits=n_split, shuffle=False)
        kf.get_n_splits(df.loc[:, df.columns != target], df[target])

    df['kfold'] = -1

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df.loc[:, df.columns != target])):
        df.loc[test_idx, 'kfold'] = fold_idx
    
    if verbose:
        print(f"Data processing completed in {time.time() - start_time:.2f} seconds")
    return df
        