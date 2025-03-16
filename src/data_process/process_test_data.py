import polars as pl
import os
import time
import sys
from typing import List, Optional

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
    # Get environment variables
    TEST_DATA = os.environ.get('TEST_DATA')
    TEST_DATA_PATH = os.environ.get('TEST_DATA_PATH')
    
    # Define columns to process
    drop_cols = ['day']
    cat_list = []
    
    # Validate required environment variables
    if TEST_DATA is None:
        print("Error: TEST_DATA environment variable is not set.")
        sys.exit(1)
        
    if TEST_DATA_PATH is None:
        print("Error: TEST_DATA_PATH environment variable is not set.")
        sys.exit(1)
    
    # Process test data
    print("Processing test data...")
    process_test_data(TEST_DATA, TEST_DATA_PATH, drop_cols, cat_list, verbose=True)
    print("Test data processing completed successfully.") 