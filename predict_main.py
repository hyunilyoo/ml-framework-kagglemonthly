import src.predict_func as pf
import polars as pl
import os
import pandas as pd

if __name__ == "__main__":
    TEST_DATA = os.environ.get('TEST_DATA')
    MODEL = os.environ.get('MODEL')
    MODEL_PATH = os.environ.get('MODEL_PATH')
    VERSION = os.environ.get('VERSION')
    TOTAL_FOLD = int(os.environ.get('TOTAL_FOLD'))
    OUTPUT_PATH = os.environ.get('OUTPUT_PATH')
    
    # Load test data
    test_df = pl.read_csv(TEST_DATA)
    
    # Convert to pandas for compatibility with predict function
    test_df = test_df.to_pandas()
    
    # Generate predictions
    predictions = pf.predict(
        test_data=test_df,
        model=MODEL,
        model_name=VERSION,
        model_path=MODEL_PATH,
        total_folds=TOTAL_FOLD
    )
    
    # Save predictions to output file
    predictions.to_csv(OUTPUT_PATH, index=False)
    print(f"Predictions saved to {OUTPUT_PATH}")
