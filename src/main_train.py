import os
import polars as pl
from . import dispatcher, train_func as t
from sklearn import metrics
import joblib

TRAINING_DATA = os.environ.get("TRAINING_DATA")
FOLD = int(os.environ.get("FOLD"))
TOTAL_FOLDS = int(os.environ.get("TOTAL_FOLDS"))
MODEL = os.environ.get("MODEL")
MODEL_FOLDER = os.environ.get("MODEL_FOLDER")
TARGET = os.environ.get("TARGET")
VERSION= os.environ.get("VERSION")
TEST_DATA = os.environ.get("TEST_DATA")

if __name__ == "__main__":
    t.clf_train(TRAINING_DATA, TARGET, MODEL, MODEL_FOLDER, FOLD, TOTAL_FOLDS, VERSION)