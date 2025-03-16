"""
Synthetic data generation module using a custom implementation of a tabular diffusion model.

This module provides functionality to generate synthetic tabular data using diffusion models,
following the TabDDPM paper (https://arxiv.org/pdf/2209.15421).
Dependencies:
- PyTorch
- pandas
- numpy
- scikit-learn
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Optional, Union, List, Dict, Tuple

# Import our custom diffusion model implementation
from tabular_diffusion import TabularDiffusionModel, generate_synthetic_data_simple_diffusion, balance_data_with_diffusion

def generate_synthetic_data_tabddpm(
    train_data: pd.DataFrame,
    categorical_columns: List[str] = None,
    numerical_columns: List[str] = None,
    num_samples: int = 1000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    epochs: int = 500,
    batch_size: int = 128,
    hidden_dims: List[int] = [512, 512, 512],
    guidance_scale: float = 3.0,
    save_model_path: Optional[str] = None,
    load_model_path: Optional[str] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic tabular data using a TabDDPM-style diffusion model.
    Implemented following the TabDDPM paper (https://arxiv.org/pdf/2209.15421).
    
    Args:
        train_data: Original training data to model
        categorical_columns: List of categorical column names
        numerical_columns: List of numerical column names
        num_samples: Number of synthetic samples to generate
        device: Device to use for training ('cuda' or 'cpu')
        epochs: Number of training epochs (500-1000 recommended for complex datasets)
        batch_size: Batch size for training
        hidden_dims: Dimensions of hidden layers in the model (larger recommended for complex data)
        guidance_scale: Classifier-free guidance scale (higher values = stronger adherence to data patterns)
        save_model_path: Path to save the trained model (optional)
        load_model_path: Path to load a pre-trained model (optional)
        seed: Random seed for reproducibility
        
    Returns:
        Dataframe containing synthetic data with the same schema as the input data
    """
    # Auto-detect categorical and numerical columns if not provided
    if categorical_columns is None and numerical_columns is None:
        categorical_columns = []
        numerical_columns = []
        for col in train_data.columns:
            if train_data[col].dtype == 'object' or pd.api.types.is_categorical_dtype(train_data[col]):
                categorical_columns.append(col)
            else:
                numerical_columns.append(col)
    
    # Convert categorical columns to category dtype if they aren't already
    for col in categorical_columns:
        if not pd.api.types.is_categorical_dtype(train_data[col]):
            train_data[col] = train_data[col].astype('category')
    
    print(f"Training TabDDPM diffusion model for {epochs} epochs on device: {device}")
    print(f"Using: {len(categorical_columns)} categorical columns, {len(numerical_columns)} numerical columns")
    print(f"Model complexity: {hidden_dims}, Guidance scale: {guidance_scale}")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create TabDDPM model with the improved architecture
    model = TabularDiffusionModel(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        hidden_dims=hidden_dims,
        guidance_scale=guidance_scale,
        device=device
    )
    
    # Load or train the model
    if load_model_path and os.path.exists(load_model_path):
        print(f"Loading model from {load_model_path}")
        model.load(load_model_path)
    else:
        print(f"Training diffusion model from scratch")
        model.fit(
            train_data=train_data,
            epochs=epochs,
            batch_size=batch_size,
            save_path=save_model_path
        )
    
    # Generate synthetic data using improved classifier-free guided sampling
    print(f"Generating {num_samples} synthetic samples with guidance scale {guidance_scale}")
    synthetic_data = model.sample(n_samples=num_samples)
    
    # Verify data types match original data
    for col in categorical_columns:
        if col in synthetic_data.columns:
            synthetic_data[col] = synthetic_data[col].astype(train_data[col].dtype)
    
    # Ensure all columns from original data exist in synthetic data
    missing_cols = set(train_data.columns) - set(synthetic_data.columns)
    if missing_cols:
        print(f"Warning: Missing columns in generated data: {missing_cols}")
        for col in missing_cols:
            print(f"Adding missing column: {col}")
            if col in categorical_columns:
                # Sample from original distribution for categorical
                values = train_data[col].sample(n=len(synthetic_data), replace=True).values
                synthetic_data[col] = values
            else:
                # For numerical columns, use original distribution
                synthetic_data[col] = np.random.normal(
                    loc=train_data[col].mean(),
                    scale=max(train_data[col].std(), 1e-6),
                    size=len(synthetic_data)
                )
    
    print(f"Generated {len(synthetic_data)} synthetic samples successfully")
    return synthetic_data


def balance_data_with_tabddpm(
    data: pd.DataFrame,
    target_column: str,
    categorical_columns: List[str] = None,
    numerical_columns: List[str] = None,
    target_ratio: Optional[Dict[any, float]] = None,
    hidden_dims: List[int] = [512, 512, 512],
    guidance_scale: float = 3.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42
) -> pd.DataFrame:
    """
    Balance imbalanced dataset using TabDDPM diffusion models to generate synthetic samples.
    
    Args:
        data: Original imbalanced data
        target_column: Name of the target column for balancing
        categorical_columns: List of categorical column names
        numerical_columns: List of numerical column names
        target_ratio: Dictionary specifying the desired ratio for each class 
                      (e.g. {0: 0.5, 1: 0.5} for binary classification)
                      If None, creates equal number of samples for each class
        hidden_dims: Dimensions of hidden layers in the model (larger recommended for complex data)
        guidance_scale: Classifier-free guidance scale (higher values = stronger adherence to data patterns)
        device: Device to use for training ('cuda' or 'cpu')
        seed: Random seed for reproducibility
        
    Returns:
        Balanced dataframe containing original and synthetic data
    """
    # Auto-detect categorical and numerical columns if not provided
    if categorical_columns is None and numerical_columns is None:
        categorical_columns = []
        numerical_columns = []
        for col in data.columns:
            if col != target_column and (data[col].dtype == 'object' or pd.api.types.is_categorical_dtype(data[col])):
                categorical_columns.append(col)
            elif col != target_column:
                numerical_columns.append(col)
    
    # Include target column in categorical columns if it's categorical
    target_is_categorical = data[target_column].dtype == 'object' or pd.api.types.is_categorical_dtype(data[target_column])
    if target_is_categorical and target_column not in categorical_columns:
        categorical_columns.append(target_column)
    elif not target_is_categorical and target_column not in numerical_columns:
        numerical_columns.append(target_column)
    
    # Convert categorical columns to category dtype if they aren't already
    for col in categorical_columns:
        if not pd.api.types.is_categorical_dtype(data[col]):
            data[col] = data[col].astype('category')
    
    # Calculate class distribution
    class_counts = data[target_column].value_counts()
    print(f"Original class distribution: {class_counts.to_dict()}")
    
    # Set target ratios if not provided
    if target_ratio is None:
        # Default to equal distribution
        unique_values = data[target_column].unique()
        target_ratio = {val: 1.0/len(unique_values) for val in unique_values}
    
    # Calculate total number of samples needed for each class
    total_samples = len(data)
    class_targets = {cls: int(ratio * total_samples) for cls, ratio in target_ratio.items()}
    
    # Generate samples for underrepresented classes
    balanced_data = data.copy()
    
    for cls, target_count in class_targets.items():
        if cls not in class_counts:
            print(f"Warning: Class {cls} not found in data")
            continue
            
        current_count = class_counts.get(cls, 0)
        
        if current_count < target_count:
            samples_needed = target_count - current_count
            print(f"Generating {samples_needed} samples for class {cls}")
            
            # Extract data for this class
            class_data = data[data[target_column] == cls]
            
            # Skip if we have no samples for this class
            if len(class_data) == 0:
                print(f"Warning: No samples for class {cls}, cannot generate synthetic data")
                continue
                
            # Generate synthetic data for this class
            synthetic_samples = generate_synthetic_data_tabddpm(
                train_data=class_data,
                categorical_columns=categorical_columns,
                numerical_columns=numerical_columns,
                num_samples=samples_needed,
                device=device,
                epochs=200,  # Fewer epochs since we have less data per class
                batch_size=min(64, len(class_data)),  # Smaller batch size for small classes
                hidden_dims=hidden_dims,
                guidance_scale=guidance_scale,
                seed=seed + hash(str(cls)) % 1000  # Different seed for each class
            )
            
            # Force the target column to have the correct class value
            synthetic_samples[target_column] = cls
            
            # Add to balanced dataset
            balanced_data = pd.concat([balanced_data, synthetic_samples], ignore_index=True)
    
    print(f"Balanced data distribution: {balanced_data[target_column].value_counts().to_dict()}")
    return balanced_data 