import pandas as pd
import os
from sklearn import model_selection
import time
import sys

def process_data(
    data_path: str, 
    save_path: str, 
    drop_cols: list[str], 
    cat_list: list[str], 
    target: str, 
    n_split: int,
    seed: int = 42,
    verbose: bool = True
) -> None:
    """
    Process and prepare data for modeling with k-fold cross-validation
    
    Args:
        data_path: Path to input CSV file
        save_path: Path to save processed CSV 
        drop_cols: List of columns to drop (can be empty)
        cat_list: List of categorical columns to convert to dummy variables (can be empty)
        target: Name of the target column
        n_split: Number of folds for cross-validation
        seed: Random seed for reproducibility
        verbose: Whether to print progress information

    Returns:
        None: Processed data is saved to save_path
    """
    start_time = time.time()
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Add k-fold column and shuffle
    if verbose:
        print(f"Data loaded and preprocessed in {time.time() - start_time:.2f} seconds")
        print(f"Dataset shape: {df.shape}")
        print("Sample data:")
        print(df.head())
        
    # Stratified k-fold split
    if df[target].dtype == 'float64' or df[target].dtype == 'float32':
        kf = model_selection.KFold(n_splits=n_split, shuffle=False)
        kf.get_n_splits(df.loc[:, df.columns != target])
    else:
        kf = model_selection.StratifiedKFold(n_splits=n_split, shuffle=False)
        kf.get_n_splits(df.loc[:, df.columns != target], df[target])

    df['kfold'] = -1

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df.loc[:, df.columns != target])):
        df.loc[test_idx, 'kfold'] = fold_idx
    
    # Save processed data
    df.to_csv(save_path, index=False)
    
    if verbose:
        print(f"Data processing completed in {time.time() - start_time:.2f} seconds")
        print(f"Processed data saved to {save_path}")
        
    return None


if __name__ == '__main__':
    drop_cols = []
    cat_list = []

    RAW_DATA = os.environ.get('RAW_DATA')
    TARGET = os.environ.get('TARGET')
    N_SPLIT = int(os.environ.get('N_SPLIT')) 
    FOLD_DATA_PATH = os.environ.get('FOLD_DATA_PATH')

    if RAW_DATA is None:
        print("Error: RAW_DATA environment variable is not set.")
        sys.exit(1)

    # Process training data if FOLD_DATA_PATH is set
    if FOLD_DATA_PATH is not None:
        if TARGET is None:
            print("Error: TARGET environment variable is not set for training data processing.")
            print("Available columns will be shown when you specify a target.")
            sys.exit(1)
        
        print("Processing training data...")
        process_data(RAW_DATA, FOLD_DATA_PATH, drop_cols, cat_list, TARGET, N_SPLIT)
    
    print("Data processing completed successfully.")