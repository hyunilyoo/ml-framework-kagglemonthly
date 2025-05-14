import src.train_func as tf
import polars as pl
import os
import numpy as np

if __name__ == "__main__":
    TRAIN_DATA = os.environ.get('TRAIN_DATA')
    TARGET = os.environ.get('TARGET')
    MODEL = os.environ.get('MODEL')
    MODEL_PATH = os.environ.get('MODEL_PATH')
    VERSION = os.environ.get('VERSION')
    EVAL_METRIC = os.environ.get('EVAL_METRIC')
    TOTAL_FOLD = int(os.environ.get('TOTAL_FOLD'))
    
    for k in range(TOTAL_FOLD):
        tf.train(TRAIN_DATA, 
                TARGET, 
                MODEL, 
                MODEL_PATH, 
                k, 
                TOTAL_FOLD, 
                VERSION, 
                EVAL_METRIC) 