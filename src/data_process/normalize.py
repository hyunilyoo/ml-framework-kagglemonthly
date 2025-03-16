import polars as pl
from sklearn.preprocessing import normalize

def normalize_data(data: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize the data using sklearn.preprocessing.normalize
    """
    return normalize(data)
