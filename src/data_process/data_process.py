import polars as pl
import os
from sklearn import model_selection
from typing import List, Optional
import time
import sys

RAW_DATA = os.environ.get('RAW_DATA')
TARGET = os.environ.get('TARGET')
N_SPLIT = int(os.environ.get('N_SPLIT')) if os.environ.get('N_SPLIT') else 5
FOLD_DATA_PATH = os.environ.get('FOLD_DATA_PATH')

drop_cols = ['id']
cat_list = []

def process_data(
    data_path: str, 
    save_path: str, 
    drop_cols: List[str], 
    cat_list: List[str], 
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
    df = pl.read_csv(data_path)
    
    # Validate target column exists
    available_columns = df.columns
    if target not in available_columns:
        print(f"Error: Target column '{target}' not found in dataset. Available columns are:")
        print(", ".join(available_columns))
        print("\nPlease set the correct TARGET environment variable.")
        sys.exit(1)
    
    # Validate drop columns exist
    if len(drop_cols) > 0:
        invalid_drop_cols = [col for col in drop_cols if col not in available_columns]
        if invalid_drop_cols:
            print(f"Warning: Some columns to drop don't exist in the dataset: {invalid_drop_cols}")
            drop_cols = [col for col in drop_cols if col in available_columns]
    
    # Validate categorical columns exist
    if len(cat_list) > 0:
        invalid_cat_cols = [col for col in cat_list if col not in available_columns]
        if invalid_cat_cols:
            print(f"Warning: Some categorical columns don't exist in the dataset: {invalid_cat_cols}")
            cat_list = [col for col in cat_list if col in available_columns]
    
    # Apply transformations conditionally
    if len(cat_list) > 0:
        df = df.to_dummies(cat_list)
        
    if len(drop_cols) > 0:
        df = df.drop(drop_cols)
    
    # Add k-fold column and shuffle
    df = (df.with_columns(kfold=pl.lit(-1))
          .sample(fraction=1, shuffle=True, seed=seed))
    
    if verbose:
        print(f"Data loaded and preprocessed in {time.time() - start_time:.2f} seconds")
        print(f"Dataset shape: {df.shape}")
        print("Sample data:")
        print(df.head())
        
    # Stratified k-fold split
    kf = model_selection.StratifiedKFold(n_splits=n_split, shuffle=False)
    
    # Create a dictionary to map fold assignments for better performance
    fold_mapping = {}
    
    # Get the indices and their fold assignments
    for fold, (_, val_idx) in enumerate(kf.split(X=df, y=df[target])):
        for idx in val_idx:
            fold_mapping[idx] = fold
            
        if verbose:
            print(f"Fold {fold}: validation set size = {len(val_idx)}")
    
    # Use with_row_count to efficiently set fold values
    df = (df.with_row_index("row_id")
          .with_columns(
              kfold=pl.when(pl.col("row_id").is_in(fold_mapping.keys()))
                    .then(pl.col("row_id").map_elements(lambda x: fold_mapping.get(x, -1), return_dtype=pl.Int32))
                    .otherwise(pl.col("kfold"))
          )
          .drop("row_id"))
    
    # Save processed data
    df.write_csv(save_path)
    
    if verbose:
        print(f"Data processing completed in {time.time() - start_time:.2f} seconds")
        print(f"Processed data saved to {save_path}")
        
    return None

def process_test_data(
    test_data_path: str,
    save_path: str,
    drop_cols: List[str],
    cat_list: List[str],
    target: Optional[str] = None,
    verbose: bool = True
) -> None:
    """
    Process test data with the same transformations as training data but without k-fold splitting
    
    Args:
        test_data_path: Path to input test CSV file
        save_path: Path to save processed test CSV
        drop_cols: List of columns to drop (can be empty)
        cat_list: List of categorical columns to convert to dummy variables (can be empty)
        target: Name of the target column (optional for test data)
        verbose: Whether to print progress information
        
    Returns:
        None: Processed test data is saved to save_path
    """
    start_time = time.time()
    
    # Load data
    df = pl.read_csv(test_data_path)
    
    # Validate columns exist
    available_columns = df.columns
    
    # Validate drop columns exist
    if len(drop_cols) > 0:
        invalid_drop_cols = [col for col in drop_cols if col not in available_columns]
        if invalid_drop_cols:
            print(f"Warning: Some columns to drop don't exist in the test dataset: {invalid_drop_cols}")
            drop_cols = [col for col in drop_cols if col in available_columns]
    
    # Validate categorical columns exist
    if len(cat_list) > 0:
        invalid_cat_cols = [col for col in cat_list if col not in available_columns]
        if invalid_cat_cols:
            print(f"Warning: Some categorical columns don't exist in the test dataset: {invalid_cat_cols}")
            cat_list = [col for col in cat_list if col in available_columns]
    
    # Apply transformations conditionally
    if len(cat_list) > 0:
        df = df.to_dummies(cat_list)
        
    if len(drop_cols) > 0:
        df = df.drop(drop_cols)
    
    if verbose:
        print(f"Test data loaded and preprocessed in {time.time() - start_time:.2f} seconds")
        print(f"Test dataset shape: {df.shape}")
        print("Sample test data:")
        print(df.head())
    
    # Save processed data
    df.write_csv(save_path)
    
    if verbose:
        print(f"Test data processing completed in {time.time() - start_time:.2f} seconds")
        print(f"Processed test data saved to {save_path}")
        
    return None

if __name__ == '__main__':
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
    
    # Process test data if TEST_DATA and TEST_DATA_PATH are set
    TEST_DATA = os.environ.get('TEST_DATA')
    TEST_DATA_PATH = os.environ.get('TEST_DATA_PATH')
    
    if TEST_DATA is not None and TEST_DATA_PATH is not None:
        print("Processing test data...")
        process_test_data(TEST_DATA, TEST_DATA_PATH, drop_cols, cat_list, TARGET, verbose=True)
    
    print("Data processing completed successfully.")