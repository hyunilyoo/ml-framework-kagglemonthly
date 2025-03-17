# ML Framework for Kaggle Monthly

A machine learning framework designed for Kaggle competitions with a focus on tabular data processing and generation using diffusion models.

## Overview

This project implements a tabular machine learning framework with the following key features:

- **Tabular Diffusion Model**: Implementation of a diffusion model for tabular data, inspired by TabDDPM
- **Data Generation**: Synthetic data generation capabilities for tabular datasets
- **Cross-Validation**: Built-in cross-validation functionality
- **Modular Design**: Separation of training, prediction, metrics, and data processing components

## Project Structure

```
├── requirements.txt       # Project dependencies
├── tabular_diffusion.py   # Tabular diffusion model implementation
├── generate_tabddpm_synthetic_data.py  # Script for synthetic data generation
├── src/                   # Source code directory
│   ├── main_train.py      # Entry point for model training
│   ├── metrics.py         # Evaluation metrics
│   ├── predict_func.py    # Prediction functionality
│   ├── train_func.py      # Training functionality
│   ├── utils.py           # Utility functions
│   ├── data_process/      # Data processing modules
│   ├── data_process.py    # Data processing utilities
│   ├── dispatcher.py      # Model dispatcher
│   ├── feature_generator.py  # Feature engineering
│   ├── loss.py            # Loss functions
│   └── cross_validation.py   # Cross-validation implementation
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training a Model

```bash
export TRAINING_DATA=path/to/train.csv
export FOLD=0
export TOTAL_FOLDS=5
export MODEL=model_name
export MODEL_FOLDER=models/
export TARGET=target_column
export VERSION=v1
python -m src.main_train
```

### Generating Synthetic Data

```bash
python generate_tabddpm_synthetic_data.py
```

## Tabular Diffusion Model

The project includes a simplified implementation of a diffusion model for tabular data inspired by TabDDPM. Key features include:

- Support for mixed categorical and numerical features
- Customizable neural network architectures
- Various diffusion schedules
- Guidance scaling for controlled generation

## Dependencies

- numpy >= 1.21.0
- pandas >= 1.3.0
- torch >= 1.9.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- polars >= 0.15.0
- joblib >= 1.1.0
- tqdm >= 4.62.0

