"""
Synthetic data generation module using TabDDPM (Tabular Denoising Diffusion Probabilistic Model).

This module provides functionality to generate synthetic tabular data using diffusion models.
Dependencies:
- PyTorch
- tabddpm (install via `pip install tabddpm`)
- pandas
- numpy
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Optional, Union, List, Dict, Tuple

def generate_synthetic_data_tabddpm(
    train_data: pd.DataFrame,
    categorical_columns: List[str] = None,
    numerical_columns: List[str] = None,
    num_samples: int = 1000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    epochs: int = 100,
    batch_size: int = 64,
    save_model_path: Optional[str] = None,
    load_model_path: Optional[str] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic tabular data using TabDDPM.
    
    Args:
        train_data: Original training data to model
        categorical_columns: List of categorical column names
        numerical_columns: List of numerical column names
        num_samples: Number of synthetic samples to generate
        device: Device to use for training ('cuda' or 'cpu')
        epochs: Number of training epochs
        batch_size: Batch size for training
        save_model_path: Path to save the trained model (optional)
        load_model_path: Path to load a pre-trained model (optional)
        seed: Random seed for reproducibility
        
    Returns:
        Dataframe containing synthetic data with the same schema as the input data
    """
    try:
        from tabddpm import TabDDPM
    except ImportError:
        raise ImportError(
            "The 'tabddpm' package is required for this function. "+
            "Please install it with: pip install tabddpm"
        )
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
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
    
    # Create a TabDDPM model
    model = TabDDPM(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns
    )
    
    # Load pre-trained model if path is provided
    if load_model_path and os.path.exists(load_model_path):
        print(f"Loading model from {load_model_path}")
        model.load_state_dict(torch.load(load_model_path))
    else:
        # Train the model on the provided data
        print(f"Training TabDDPM model for {epochs} epochs on device: {device}")
        model.fit(
            train_data=train_data,
            epochs=epochs,
            batch_size=batch_size,
            device=device
        )
        
        # Save model if path is provided
        if save_model_path:
            os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
            torch.save(model.state_dict(), save_model_path)
            print(f"Model saved to {save_model_path}")
    
    # Generate synthetic data
    print(f"Generating {num_samples} synthetic samples")
    synthetic_data = model.sample(n_samples=num_samples)
    
    # Verify data types match original data
    for col in categorical_columns:
        if col in synthetic_data.columns:
            synthetic_data[col] = synthetic_data[col].astype(train_data[col].dtype)
    
    print(f"Generated {len(synthetic_data)} synthetic samples")
    return synthetic_data


def balance_data_with_tabddpm(
    data: pd.DataFrame,
    target_column: str,
    categorical_columns: List[str] = None,
    numerical_columns: List[str] = None,
    target_ratio: Optional[Dict[any, float]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42
) -> pd.DataFrame:
    """
    Balance imbalanced dataset using TabDDPM to generate synthetic samples.
    
    Args:
        data: Original imbalanced data
        target_column: Name of the target column for balancing
        categorical_columns: List of categorical column names
        numerical_columns: List of numerical column names
        target_ratio: Dictionary specifying the desired ratio for each class 
                      (e.g. {0: 0.5, 1: 0.5} for binary classification)
                      If None, creates equal number of samples for each class
        device: Device to use for training ('cuda' or 'cpu')
        seed: Random seed for reproducibility
        
    Returns:
        Balanced dataframe containing original and synthetic data
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Get class distribution
    class_counts = data[target_column].value_counts()
    majority_class = class_counts.idxmax()
    majority_count = class_counts[majority_class]
    
    # Print initial class distribution for debugging
    print(f"Initial class distribution: {dict(class_counts)}")
    
    # Determine target counts for each class
    if target_ratio is None:
        # Equal distribution if not specified
        target_counts = {cls: majority_count for cls in class_counts.index}
    else:
        # Calculate based on provided ratios
        total_samples = sum(target_ratio.values())
        normalized_ratios = {k: v/total_samples for k, v in target_ratio.items()}
        
        # Determine total number of samples needed
        max_class = max(normalized_ratios.items(), key=lambda x: x[1])[0]
        max_ratio = normalized_ratios[max_class]
        max_count = class_counts[max_class]
        total_desired = int(max_count / max_ratio)
        
        # Calculate target counts for each class
        target_counts = {cls: int(total_desired * normalized_ratios.get(cls, 0)) 
                         for cls in class_counts.index}
    
    print(f"Target counts per class: {target_counts}")
    
    # Initialize the result with an empty DataFrame to build from scratch
    balanced_data = pd.DataFrame(columns=data.columns)
    
    # Make sure numerical_columns and categorical_columns are properly initialized
    if numerical_columns is None:
        numerical_columns = []
        for col in data.columns:
            if col != target_column and not (data[col].dtype == 'object' or pd.api.types.is_categorical_dtype(data[col])):
                numerical_columns.append(col)
    
    if categorical_columns is None:
        categorical_columns = []
        for col in data.columns:
            if col != target_column and (data[col].dtype == 'object' or pd.api.types.is_categorical_dtype(data[col])):
                categorical_columns.append(col)
    
    # Explicitly filter out the target column from feature lists
    if isinstance(categorical_columns, list) and target_column in categorical_columns:
        categorical_columns = [col for col in categorical_columns if col != target_column]
    
    if isinstance(numerical_columns, list) and target_column in numerical_columns:
        numerical_columns = [col for col in numerical_columns if col != target_column]
        
    print(f"Using categorical columns: {categorical_columns}")
    print(f"Using numerical columns: {numerical_columns}")
    
    # Process each class to achieve target distribution
    for cls in class_counts.index:
        current_count = class_counts[cls]
        target_count = target_counts[cls]
        
        # Get samples for this class
        class_data = data[data[target_column] == cls].copy()
        
        if current_count > target_count:
            # Downsample overrepresented classes
            print(f"Downsampling class {cls} from {current_count} to {target_count} samples")
            sampled_class_data = class_data.sample(n=target_count, random_state=seed)
            balanced_data = pd.concat([balanced_data, sampled_class_data], ignore_index=True)
            
        elif current_count < target_count:
            # Generate synthetic samples for underrepresented classes
            samples_needed = target_count - current_count
            print(f"Generating {samples_needed} synthetic samples for class {cls}")
            
            # Add all original samples from this class
            balanced_data = pd.concat([balanced_data, class_data], ignore_index=True)
            
            # Generate synthetic samples using TabDDPM
            synthetic_samples = generate_synthetic_data_tabddpm(
                train_data=class_data,
                categorical_columns=categorical_columns,
                numerical_columns=numerical_columns,
                num_samples=samples_needed,
                device=device,
                epochs=50,  # Reduced epochs for faster processing
                batch_size=32,
                seed=seed
            )
            
            # Ensure all columns from original data exist in synthetic data
            for col in data.columns:
                if col not in synthetic_samples.columns:
                    print(f"Adding missing column: {col}")
                    # For categorical columns, sample from original distribution
                    if col in categorical_columns or data[col].dtype == 'object' or pd.api.types.is_categorical_dtype(data[col]):
                        values = class_data[col].sample(n=len(synthetic_samples), replace=True).values
                        synthetic_samples[col] = values
                    # For numerical columns, use mean or 0
                    else:
                        synthetic_samples[col] = class_data[col].mean() if len(class_data) > 0 else 0
            
            # Force the target column to the correct class value
            synthetic_samples[target_column] = cls
            
            # Combine with balanced data
            balanced_data = pd.concat([balanced_data, synthetic_samples], ignore_index=True)
        else:
            # Class already has the exact target count, just add all samples
            balanced_data = pd.concat([balanced_data, class_data], ignore_index=True)
    
    # Add a verification step to ensure the final distribution matches the target
    final_distribution = balanced_data[target_column].value_counts().to_dict()
    print(f"Original class distribution: {dict(class_counts)}")
    print(f"Target class distribution: {target_counts}")
    print(f"Final class distribution: {final_distribution}")
    
    # Verify the dataset is balanced as expected
    for cls, target in target_counts.items():
        actual = final_distribution.get(cls, 0)
        if abs(actual - target) > 5:  # Allow small differences due to rounding
            print(f"WARNING: Class {cls} has {actual} samples but target was {target}")
    
    return balanced_data
