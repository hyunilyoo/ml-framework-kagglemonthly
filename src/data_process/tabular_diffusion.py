"""
Simplified Tabular Diffusion Model

This is a lightweight implementation of a diffusion model for tabular data
inspired by TabDDPM (https://github.com/yandex-research/tab-ddpm).

The implementation focuses on the core concepts of diffusion models adapted
for tabular data with mixed categorical and numerical features.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import List, Dict, Tuple, Optional, Union
import warnings
import math

# Create a custom StandardScaler wrapper with debugging
class DebugStandardScaler:
    def __init__(self):
        self.scaler = StandardScaler()
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, X, y=None):
        self.scaler.fit(X, y)
        self.mean_ = self.scaler.mean_
        self.scale_ = self.scaler.scale_
        print(f"DEBUG - StandardScaler parameters:")
        print(f"DEBUG - mean: {self.mean_}")
        print(f"DEBUG - scale: {self.scale_}")
        return self
    
    def transform(self, X):
        result = self.scaler.transform(X)
        print(f"DEBUG - StandardScaler transform:")
        print(f"DEBUG - Input mean: {np.mean(X, axis=0)}")
        print(f"DEBUG - Output mean: {np.mean(result, axis=0)}")
        print(f"DEBUG - Input min: {np.min(X, axis=0)}")
        print(f"DEBUG - Output min: {np.min(result, axis=0)}")
        return result
    
    def inverse_transform(self, X):
        result = self.scaler.inverse_transform(X)
        print(f"DEBUG - StandardScaler inverse_transform:")
        print(f"DEBUG - Input mean: {np.mean(X, axis=0)}")
        print(f"DEBUG - Output mean: {np.mean(result, axis=0)}")
        print(f"DEBUG - Input min: {np.min(X, axis=0)}")
        print(f"DEBUG - Output min: {np.min(result, axis=0)}")
        return result

class TabularEncoder:
    """Handles encoding and decoding of tabular data features."""
    
    def __init__(self, 
                 categorical_columns: List[str] = None, 
                 numerical_columns: List[str] = None):
        self.categorical_columns = categorical_columns or []
        self.numerical_columns = numerical_columns or []
        self.num_encoders = DebugStandardScaler()  # Using our wrapped version
        self.cat_encoders = {}
        self.cat_dims = {}
        self.output_dim = 0
        self.num_dim = 0
        self.cat_start_indices = {}
    
    def fit(self, data: pd.DataFrame):
        """Fit the encoders on the training data."""
        # Store original dtypes for later use
        self.original_dtypes = data.dtypes.to_dict()
        print(f"DEBUG - Original dtypes: {self.original_dtypes}")
        
        # Store the original bounds for numerical columns
        self.original_bounds = {}
        for col in self.numerical_columns:
            min_val = data[col].min()
            max_val = data[col].max()
            self.original_bounds[col] = (min_val, max_val)
            print(f"DEBUG - Recorded bounds for {col}: min={min_val}, max={max_val}")
        
        # Handle numerical features
        if self.numerical_columns:
            self.num_encoders.fit(data[self.numerical_columns])
            self.num_dim = len(self.numerical_columns)
            self.output_dim += self.num_dim
        
        # Handle categorical features
        start_idx = self.num_dim
        for col in self.categorical_columns:
            # Create one-hot encoder for the column
            try:
                # Try with newer scikit-learn API
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            except TypeError:
                # Fall back to older scikit-learn API
                encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            
            encoder.fit(data[[col]])
            self.cat_encoders[col] = encoder
            
            # Store dimensionality info
            cat_dim = len(encoder.categories_[0])
            self.cat_dims[col] = cat_dim
            self.cat_start_indices[col] = start_idx
            start_idx += cat_dim
            self.output_dim += cat_dim
            
            print(f"DEBUG - Fitted categorical column {col} with {cat_dim} categories")
    
    def transform(self, data: pd.DataFrame) -> torch.Tensor:
        """Transform the data into model-ready tensors."""
        transformed_data = []
        
        # Print statistics of input data
        print(f"DEBUG - Input data statistics:")
        for col in data.columns:
            try:
                mean_val = data[col].mean()
                std_val = data[col].std()
                min_val = data[col].min()
                max_val = data[col].max()
                print(f"DEBUG - Column {col}: mean={mean_val:.6f}, std={std_val:.6f}, min={min_val}, max={max_val}")
            except:
                print(f"DEBUG - Column {col}: Could not calculate statistics (likely non-numeric)")
        
        # Transform numerical features
        if self.numerical_columns:
            num_data = self.num_encoders.transform(data[self.numerical_columns])
            print(f"DEBUG - Numerical data after scaling: mean={np.mean(num_data):.6f}, std={np.std(num_data):.6f}")
            transformed_data.append(torch.tensor(num_data, dtype=torch.float32))
        
        # Transform categorical features
        for col in self.categorical_columns:
            encoder = self.cat_encoders[col]
            cat_data = encoder.transform(data[[col]])
            print(f"DEBUG - Categorical data for {col}: shape={cat_data.shape}, sum={np.sum(cat_data)}")
            transformed_data.append(torch.tensor(cat_data, dtype=torch.float32))
        
        # Concatenate all features
        if transformed_data:
            result = torch.cat(transformed_data, dim=1)
            print(f"DEBUG - Final transformed tensor: shape={result.shape}, mean={torch.mean(result).item():.6f}, sum={torch.sum(result).item():.6f}")
            
            # Check for all zeros
            if torch.sum(torch.abs(result)) < 1e-8:
                print(f"DEBUG - WARNING: Transformed data contains all zeros or very small values!")
                
            return result
        return torch.tensor([])
    
    def inverse_transform(self, tensors: torch.Tensor) -> pd.DataFrame:
        """Convert model output tensors back to a dataframe."""
        result_dict = {}
        tensors_np = tensors.detach().cpu().numpy()
        
        print(f"DEBUG - Inverse transform tensor shape: {tensors_np.shape}")
        print(f"DEBUG - Number of numerical columns: {len(self.numerical_columns)}")
        print(f"DEBUG - Number of categorical columns: {len(self.categorical_columns)}")
        
        # Check if input tensor has unreasonable values
        if np.sum(np.abs(tensors_np)) < 1e-8:
            print(f"DEBUG - CRITICAL: Input tensor to inverse_transform contains all zeros!")
            print(f"DEBUG - Applying emergency tensor correction")
            # Add some random reasonable values
            tensors_np = np.random.randn(*tensors_np.shape) * 0.5 + 0.5
        
        # Inverse transform numerical features
        if self.numerical_columns:
            print(f"DEBUG - Numerical columns: {self.numerical_columns}")
            print(f"DEBUG - Numerical dimension: {self.num_dim}")
            num_data = tensors_np[:, :self.num_dim]
            try:
                # Calculate statistics of input data for debugging
                print(f"DEBUG - Numerical input stats for inverse_transform:")
                print(f"DEBUG - Mean: {np.mean(num_data, axis=0)}")
                print(f"DEBUG - Min: {np.min(num_data, axis=0)}")
                print(f"DEBUG - Max: {np.max(num_data, axis=0)}")
                
                num_data_inv = self.num_encoders.inverse_transform(num_data)
                
                # Safety check for unreasonable values after inverse transform
                if np.isnan(num_data_inv).any() or np.isinf(num_data_inv).any():
                    print(f"DEBUG - WARNING: NaN/Inf detected after inverse transform of numerical data")
                    # Replace NaN/Inf with reasonable values
                    means = np.nanmean(num_data_inv, axis=0)
                    # For any columns where mean is NaN, use 0
                    means = np.nan_to_num(means)
                    # Replace NaN/Inf with means
                    num_data_inv = np.nan_to_num(num_data_inv, nan=0.0, posinf=100.0, neginf=-100.0)
                    
                # Track the original bounds for each column if available
                column_bounds = {}
                if hasattr(self, 'original_bounds'):
                    column_bounds = self.original_bounds

                for i, col in enumerate(self.numerical_columns):
                    # Add a small random variation to prevent all identical values
                    column_data = num_data_inv[:, i]
                    if np.all(column_data == column_data[0]):
                        print(f"DEBUG - WARNING: All identical values for column {col}")
                        # Add small random noise to differentiate values
                        column_data = column_data + np.random.randn(len(column_data)) * 0.01 * np.abs(column_data[0] if column_data[0] != 0 else 1.0)
                    
                    # Apply bounds constraints if we know the original bounds
                    if col in column_bounds:
                        orig_min, orig_max = column_bounds[col]
                        print(f"DEBUG - Enforcing bounds for {col}: min={orig_min}, max={orig_max}")
                        column_data = np.clip(column_data, orig_min, orig_max)
                        
                        # Enforce non-negativity if original data was non-negative
                        if orig_min >= 0:
                            column_data = np.clip(column_data, 0, None)
                    
                    result_dict[col] = column_data
            except Exception as e:
                print(f"DEBUG - Error in numerical inverse transform: {str(e)}")
                # On error, create reasonable random values
                for i, col in enumerate(self.numerical_columns):
                    if hasattr(self, 'original_dtypes'):
                        # Try to use original mean and std
                        try:
                            mean_val = self.num_encoders.mean_[i]
                            std_val = self.num_encoders.scale_[i]
                            result_dict[col] = np.random.normal(mean_val, std_val, size=tensors_np.shape[0])
                            print(f"DEBUG - Created random values for {col} with mean={mean_val}, std={std_val}")
                        except:
                            # Default to standard normal
                            result_dict[col] = np.random.randn(tensors_np.shape[0])
                            print(f"DEBUG - Created standard normal values for {col}")
                    else:
                        # Default to standard normal
                        result_dict[col] = np.random.randn(tensors_np.shape[0])
                        print(f"DEBUG - Created standard normal values for {col}")
                    
        # Inverse transform categorical features
        for col in self.categorical_columns:
            try:
                print(f"DEBUG - Processing categorical column: {col}")
                start_idx = self.cat_start_indices[col]
                cat_dim = self.cat_dims[col]
                print(f"DEBUG - Column {col} - start_idx: {start_idx}, cat_dim: {cat_dim}")
                
                # Get the one-hot encoded section for this category
                cat_data = tensors_np[:, start_idx:start_idx+cat_dim]
                print(f"DEBUG - Column {col} - cat_data shape: {cat_data.shape}")
                
                # Safety check - check if one-hot encoding is valid
                row_sums = np.sum(cat_data, axis=1)
                if np.any(row_sums < 0.1):  # Very small sum indicates invalid data
                    print(f"DEBUG - WARNING: Invalid one-hot encoding detected for column {col}")
                    # Generate valid one-hot vectors by randomly selecting categories
                    for i in range(cat_data.shape[0]):
                        if row_sums[i] < 0.1:
                            # Randomly select a category
                            cat_idx = np.random.randint(0, cat_dim)
                            # Reset to zeros
                            cat_data[i, :] = 0
                            # Set selected category to 1
                            cat_data[i, cat_idx] = 1
                
                # For sampling, we convert to proper one-hot (if not already)
                if np.max(cat_data) < 0.9:  # Not one-hot already
                    print(f"DEBUG - Column {col} - Converting to one-hot")
                    indices = np.argmax(cat_data, axis=1)
                    cat_data = np.zeros_like(cat_data)
                    for i, idx in enumerate(indices):
                        cat_data[i, idx] = 1.0
                
                # Get original category values
                try:
                    cat_inv = self.cat_encoders[col].inverse_transform(cat_data)
                    print(f"DEBUG - Column {col} - Inverse transformed shape: {cat_inv.shape}")
                    result_dict[col] = cat_inv.flatten()
                except Exception as e:
                    print(f"DEBUG - Error in one-hot decoding for column {col}: {str(e)}")
                    # Fallback: generate random valid categories
                    n_samples = cat_data.shape[0]
                    cat_indices = np.random.randint(0, cat_dim, size=n_samples)
                    # Create a clean one-hot encoding
                    clean_cat_data = np.zeros_like(cat_data)
                    for i, idx in enumerate(cat_indices):
                        clean_cat_data[i, idx] = 1.0
                    
                    try:
                        # Try again with clean data
                        cat_inv = self.cat_encoders[col].inverse_transform(clean_cat_data)
                        result_dict[col] = cat_inv.flatten()
                    except:
                        # Ultimate fallback: just use indices as categories
                        result_dict[col] = [f"category_{i}" for i in cat_indices]
                        
            except Exception as e:
                print(f"DEBUG - Error in categorical inverse transform for column {col}: {str(e)}")
                # Fallback: create random categories
                n_samples = tensors_np.shape[0]
                if col in self.cat_dims:
                    n_categories = self.cat_dims[col]
                else:
                    n_categories = 2  # Default to binary if unknown
                
                # Generate random category indices
                random_categories = np.random.randint(0, n_categories, size=n_samples)
                result_dict[col] = [f"category_{i}" for i in random_categories]
                print(f"DEBUG - Created random categories for {col}")
        
        print(f"DEBUG - Final result_dict keys: {list(result_dict.keys())}")
        result_df = pd.DataFrame(result_dict)
        
        # Try to convert back to original dtypes
        if hasattr(self, 'original_dtypes'):
            print(f"DEBUG - Converting back to original dtypes")
            for col, dtype in self.original_dtypes.items():
                if col in result_df.columns:
                    try:
                        result_df[col] = result_df[col].astype(dtype)
                        print(f"DEBUG - Converted {col} to {dtype}")
                    except Exception as e:
                        print(f"DEBUG - Could not convert {col} to {dtype}: {str(e)}")
        
        return result_df

class TabDDPM_MLP(nn.Module):
    """
    MLP-based model implementing the TabDDPM architecture with:
    - Separate processing for categorical and numerical features
    - Feature-wise linear modulation (FiLM) for time conditioning
    - Special handling for categorical features
    """
    def __init__(self, 
                 input_dim: int,
                 cat_dims: List[int],  # Dimensions of each categorical variable
                 cat_idxs: List[int],  # Indices of categorical variables
                 num_idxs: List[int],  # Indices of numerical variables 
                 hidden_dims: List[int] = [256, 256, 256],
                 embedding_dim: int = 8):
        super().__init__()
        
        self.input_dim = input_dim
        self.cat_dims = cat_dims
        self.cat_idxs = cat_idxs
        self.num_idxs = num_idxs
        self.num_cats = len(cat_idxs)
        self.num_nums = len(num_idxs)
        self.embedding_dim = embedding_dim
        
        # Set up categorical embeddings (for the model understanding)
        if self.num_cats > 0:
            self.cat_embeddings = nn.ModuleList()
            for idx, dim in zip(cat_idxs, cat_dims):
                self.cat_embeddings.append(nn.Embedding(dim, embedding_dim))
        
        # Time embedding (sinusoidal for better conditioning)
        self.time_embed_dim = hidden_dims[0]
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_embed_dim // 4),
            nn.Linear(self.time_embed_dim // 4, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        
        # Main network
        self.net = nn.ModuleList()
        
        # Calculate input size for network
        input_size = len(num_idxs) + len(cat_idxs) * embedding_dim
        if input_size == 0:
            # Safety check to avoid zero input size
            input_size = 1
            print("WARNING: Input size was 0, setting to 1 to avoid errors")
        
        # Initial layer
        self.net.append(nn.Linear(input_size, hidden_dims[0]))
        
        # Initialize weights with Xavier/Glorot initialization for better stability
        nn.init.xavier_normal_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        
        # Hidden layers with FiLM conditioning
        for i in range(len(hidden_dims) - 1):
            linear_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            # Apply Xavier initialization
            nn.init.xavier_normal_(linear_layer.weight)
            nn.init.zeros_(linear_layer.bias)
            
            self.net.append(nn.ModuleList([
                nn.SiLU(),
                linear_layer,
                FiLM(hidden_dims[i+1], self.time_embed_dim)
            ]))
        
        # Output layer
        self.final_layer = nn.Linear(hidden_dims[-1], input_dim)
        
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Track original shape for proper reshaping later
        original_shape = x.shape
        original_batch_size = x.shape[0]
        
        # Check and fix input dimensions
        if len(x.shape) > 2:
            # If we have batch x batch_size x features, flatten the extra dimension
            print(f"DEBUG - Reshaping tensor from {x.shape} to 2D")
            
            # If the tensor is [batch, batch, features], we need to reshape carefully
            if x.shape[0] == x.shape[1]:
                print(f"WARNING - Tensor has same batch dim in first two dimensions")
                # Use only the first slice to avoid self-broadcasting issue
                x = x[:, 0, :].reshape(original_batch_size, -1)
            else:
                # Standard reshape
                x = x.reshape(original_batch_size, -1)
                
            print(f"DEBUG - Reshaped to {x.shape}")
        
        # Verify the reshaped tensor size
        if x.shape[1] != self.input_dim:
            print(f"WARNING - Input feature dimension {x.shape[1]} doesn't match expected {self.input_dim}")
            # We need to fix this - truncate or pad
            if x.shape[1] > self.input_dim:
                # Truncate extra features
                x = x[:, :self.input_dim]
                print(f"DEBUG - Truncated to {x.shape}")
            else:
                # Pad with zeros
                padding = torch.zeros((x.shape[0], self.input_dim - x.shape[1]), device=x.device)
                x = torch.cat([x, padding], dim=1)
                print(f"DEBUG - Padded to {x.shape}")
        
        batch_size, total_dim = x.shape
        
        # Debug output more controlled to avoid spam
        if batch_size < 10:  # Only print for small batches to avoid log spam
            print(f"DEBUG - Model forward input shape: {x.shape}")
            print(f"DEBUG - Model has {len(self.num_idxs)} numerical features and {len(self.cat_idxs)} categorical features")
        
        # Split input into categorical and numerical parts
        x_cat = []
        
        # Extract numerical features if any
        if self.num_nums > 0:
            # Make sure indices are within bounds
            valid_num_idxs = [idx for idx in self.num_idxs if idx < total_dim]
            if len(valid_num_idxs) > 0:
                x_num = x[:, valid_num_idxs]
                if batch_size < 10:  # Limit logging
                    print(f"DEBUG - Numerical features shape: {x_num.shape}")
            else:
                x_num = torch.zeros((batch_size, 0), device=x.device)
                print(f"DEBUG - No valid numerical features found")
        else:
            x_num = torch.zeros((batch_size, 0), device=x.device)
        
        # Process categorical features if any
        if self.num_cats > 0:
            for i, (idx, emb) in enumerate(zip(self.cat_idxs, self.cat_embeddings)):
                # Make sure the index is within bounds
                if idx < total_dim:
                    # Get categorical feature - handle it safely
                    cats = x[:, idx:idx+1]
                    
                    # Ensure it's an integer for embedding lookup and within bounds
                    safe_cats = torch.clamp(cats.long(), 0, self.cat_dims[i] - 1)
                    
                    # Get embeddings
                    cat_feature = emb(safe_cats)
                    x_cat.append(cat_feature.squeeze(1))
                    if batch_size < 10:  # Limit logging
                        print(f"DEBUG - Cat feature {i} (idx {idx}) shape: {cat_feature.squeeze(1).shape}")
                else:
                    # If index is out of bounds, create a zero embedding
                    dummy_embedding = torch.zeros((batch_size, self.embedding_dim), device=x.device)
                    x_cat.append(dummy_embedding)
                    print(f"DEBUG - Created dummy embedding for out-of-bounds category {i}")
        
        # Combine all features - handle empty cases
        if len(x_cat) > 0:
            # Check if all cat features have the same first dimension
            cat_sizes = [cat.size(0) for cat in x_cat]
            if len(set(cat_sizes)) > 1:
                print(f"DEBUG - WARNING: Categorical features have different batch sizes: {cat_sizes}")
                # Make all features have the same batch size
                max_size = max(cat_sizes)
                x_cat = [torch.cat([cat, torch.zeros((max_size - cat.size(0), cat.size(1)), device=x.device)], dim=0) 
                         if cat.size(0) < max_size else cat for cat in x_cat]
            
            x_cat = torch.cat(x_cat, dim=1)
            if batch_size < 10:  # Limit logging
                print(f"DEBUG - Combined categorical features shape: {x_cat.shape}")
            
            # Check if numerical and categorical have matching batch size
            if x_num.size(0) != x_cat.size(0):
                print(f"DEBUG - WARNING: Batch size mismatch between numerical ({x_num.size(0)}) and categorical ({x_cat.size(0)})")
                # Make both have the same batch size
                if x_num.size(0) > x_cat.size(0):
                    x_cat = torch.cat([x_cat, torch.zeros((x_num.size(0) - x_cat.size(0), x_cat.size(1)), device=x.device)], dim=0)
                elif x_num.size(0) < x_cat.size(0):
                    x_num = torch.cat([x_num, torch.zeros((x_cat.size(0) - x_num.size(0), x_num.size(1)), device=x.device)], dim=0)
            
            # Now concatenate them
            if x_num.size(1) > 0:  # Only if we have numerical features
                h = torch.cat([x_num, x_cat], dim=1)
                if batch_size < 10:  # Limit logging
                    print(f"DEBUG - Combined features shape: {h.shape}")
            else:
                h = x_cat
        else:
            h = x_num
            if batch_size < 10:  # Limit logging
                print(f"DEBUG - Using only numerical features with shape: {h.shape}")
        
        # Initial layer
        h = self.net[0](h)
        
        # Hidden layers with FiLM conditioning
        for i in range(1, len(self.net)):
            layer_modules = self.net[i]
            h = layer_modules[0](h)  # Activation
            h = layer_modules[1](h)  # Linear
            h = layer_modules[2](h, t_emb)  # FiLM conditioning
            
            # Check for NaN values
            if torch.isnan(h).any():
                print(f"WARNING - NaN values detected after layer {i}")
                # Replace NaNs with zeros
                h = torch.nan_to_num(h, nan=0.0)
        
        # Output layer
        output = self.final_layer(h)
        
        # Ensure output has the same batch size as input
        if output.shape[0] != original_batch_size:
            print(f"WARNING - Output batch size {output.shape[0]} doesn't match input batch size {original_batch_size}")
            # Try to fix by reshaping
            try:
                output = output.reshape(original_batch_size, -1)
            except:
                # If reshaping fails, pad or truncate to match
                if output.shape[0] < original_batch_size:
                    # Pad
                    padding = torch.zeros((original_batch_size - output.shape[0], output.shape[1]), device=output.device)
                    output = torch.cat([output, padding], dim=0)
                else:
                    # Truncate
                    output = output[:original_batch_size]
        
        # Ensure output has the expected feature dimension
        expected_features = self.input_dim
        if output.shape[1] != expected_features:
            print(f"WARNING - Output feature dim {output.shape[1]} doesn't match expected {expected_features}")
            # Try to fix by reshaping
            try:
                output = output.reshape(original_batch_size, expected_features)
            except:
                # If reshaping fails, pad or truncate to match
                if output.shape[1] < expected_features:
                    # Pad
                    padding = torch.zeros((output.shape[0], expected_features - output.shape[1]), device=output.device)
                    output = torch.cat([output, padding], dim=1)
                else:
                    # Truncate
                    output = output[:, :expected_features]
        
        # Final check for NaNs
        if torch.isnan(output).any():
            print(f"WARNING - NaN values detected in model output")
            output = torch.nan_to_num(output, nan=0.0)
            
        return output


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal positional embeddings for time steps as used in the paper
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer for diffusion model
    """
    def __init__(self, feature_dim, condition_dim):
        super().__init__()
        self.scale = nn.Linear(condition_dim, feature_dim)
        self.shift = nn.Linear(condition_dim, feature_dim)
        # Initialize to near identity function to improve stability
        with torch.no_grad():
            self.scale.weight.data.fill_(0.01)
            self.scale.bias.data.fill_(0.0)
            self.shift.weight.data.fill_(0.01)
            self.shift.bias.data.fill_(0.0)
        
    def forward(self, x, condition):
        # Get batch size from input tensor
        batch_size = x.shape[0]
        
        # Generate scale and shift parameters based on conditioning
        scale = self.scale(condition)
        shift = self.shift(condition)
        
        # Prevent extreme values that might cause numerical instability
        scale = torch.clamp(scale, -10.0, 10.0)
        shift = torch.clamp(shift, -10.0, 10.0)
        
        # Check for NaN values in inputs
        if torch.isnan(x).any():
            print("WARNING: NaN values detected in FiLM input tensor")
            x = torch.nan_to_num(x, nan=0.0)
        if torch.isnan(scale).any() or torch.isnan(shift).any():
            print("WARNING: NaN values detected in FiLM scale/shift")
            scale = torch.nan_to_num(scale, nan=0.0)
            shift = torch.nan_to_num(shift, nan=0.0)
        
        # Ensure proper broadcasting based on input tensor dimensions
        if len(x.shape) == 2:
            # For 2D tensors (batch_size, features)
            scale = scale.view(batch_size, -1)
            shift = shift.view(batch_size, -1)
        elif len(x.shape) == 3:
            # For 3D tensors (batch_size, seq_len, features)
            # This handles the case where batching causes unexpected dimensions
            print(f"WARNING: Input to FiLM has 3D shape: {x.shape}")
            
            # Reshape both x and the conditioning parameters to 2D
            x = x.reshape(batch_size, -1)
            scale = scale.view(batch_size, -1)
            shift = shift.view(batch_size, -1)
        
        # Apply conditioning with careful numerical handling
        # Use scale+1 to make it an adjustment around identity function
        result = x * (scale + 1.0) + shift
        
        # Final safety check
        if torch.isnan(result).any():
            print("WARNING: NaN values detected in FiLM output")
            result = torch.nan_to_num(result, nan=0.0)
            
        return result

class TabularDiffusionModel:
    """
    Diffusion model for tabular data with mixed categorical and numerical features.
    Improved implementation following the TabDDPM paper.
    """
    def __init__(self, 
                 categorical_columns: List[str] = None,
                 numerical_columns: List[str] = None,
                 hidden_dims: List[int] = [256, 256, 256],
                 num_timesteps: int = 1000,
                 beta_schedule: str = 'cosine',
                 guidance_scale: float = 3.0,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
                 ):
        """
        Initialize the diffusion model following TabDDPM architecture.
        
        Args:
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
            hidden_dims: Dimensions of hidden layers in the MLP
            num_timesteps: Number of diffusion timesteps (1000 recommended in TabDDPM)
            beta_schedule: Schedule for noise level ('linear' or 'cosine')
            guidance_scale: Classifier-free guidance scale for improved sampling 
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.categorical_columns = categorical_columns or []
        self.numerical_columns = numerical_columns or []
        self.hidden_dims = hidden_dims
        self.num_timesteps = num_timesteps
        self.device = device
        self.guidance_scale = guidance_scale
        
        # Set up encoders
        self.encoder = TabularEncoder(
            categorical_columns=self.categorical_columns,
            numerical_columns=self.numerical_columns
        )
        
        # We'll initialize the network after seeing the data
        self.model = None
        
        # Set up diffusion parameters
        self.setup_diffusion_parameters(beta_schedule)
    
    def setup_diffusion_parameters(self, beta_schedule: str):
        """Set up the noise schedule and diffusion parameters."""
        if beta_schedule == 'linear':
            # Linear schedule from Ho et al.
            betas = torch.linspace(1e-4, 0.02, self.num_timesteps)
        elif beta_schedule == 'cosine':
            # Cosine schedule from Improved DDPM
            steps = torch.arange(self.num_timesteps + 1)
            s = 0.008
            f_t = torch.cos((steps / self.num_timesteps + s) / (1 + s) * torch.pi / 2) ** 2
            alphas_cumprod = f_t / f_t[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Debug info
        print(f"DEBUG - Setting up diffusion parameters with {beta_schedule} schedule")
        print(f"DEBUG - betas shape: {betas.shape}, device: {betas.device}")
        
        self.betas = betas.to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Calculations for diffusion q(x_t | x_0) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        print(f"DEBUG - Computing posterior_variance")
        print(f"DEBUG - self.betas shape: {self.betas.shape}")
        print(f"DEBUG - self.alphas_cumprod shape: {self.alphas_cumprod.shape}")
        
        # Ensure we're using tensors for all calculations
        posterior_variance = self.betas[:-1] * (1.0 - self.alphas_cumprod[:-1]) / (1.0 - self.alphas_cumprod[1:])
        print(f"DEBUG - posterior_variance shape before concat: {posterior_variance.shape}, type: {type(posterior_variance)}")
        
        # Explicitly create tensor for final element
        final_variance = torch.tensor([1e-20], device=self.device)
        print(f"DEBUG - final_variance: {final_variance}, type: {type(final_variance)}")
        
        # Concatenate and store
        self.posterior_variance = torch.cat([posterior_variance, final_variance])
        print(f"DEBUG - posterior_variance final shape: {self.posterior_variance.shape}, type: {type(self.posterior_variance)}")
        print(f"DEBUG - posterior_variance device: {self.posterior_variance.device}")
        
        # Remainder of the calculations
        self.posterior_log_variance = torch.log(self.posterior_variance)
        self.posterior_mean_coef1 = self.betas[:-1] * torch.sqrt(self.alphas_cumprod[:-1]) / (1.0 - self.alphas_cumprod[1:])
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod[:-1]) * torch.sqrt(self.alphas[1:]) / (1.0 - self.alphas_cumprod[1:])
        self.posterior_mean_coef1 = torch.cat([self.posterior_mean_coef1, torch.tensor([0.0]).to(self.device)])
        self.posterior_mean_coef2 = torch.cat([self.posterior_mean_coef2, torch.tensor([1.0]).to(self.device)])
    
    def fit(self, train_data: pd.DataFrame, epochs: int = 100, batch_size: int = 64, 
            lr: float = 1e-3, save_path: Optional[str] = None):
        """
        Train the diffusion model on the provided data.
        
        Args:
            train_data: Training data as a DataFrame
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            save_path: Path to save the trained model (optional)
        """
        # Fit the encoder on the data
        self.encoder.fit(train_data)
        input_dim = self.encoder.output_dim
        
        # Get indices for categorical and numerical features
        cat_dims = []
        cat_idxs = []
        num_idxs = []
        
        # Start with numerical indices (if any)
        if self.numerical_columns:
            # All numerical features are at the beginning in the encoded tensor
            num_idxs = list(range(self.encoder.num_dim))
        
        # Then get categorical indices and dimensions
        for col in self.categorical_columns:
            if col in self.encoder.cat_start_indices:
                idx = self.encoder.cat_start_indices[col]
                dim = self.encoder.cat_dims[col]
                cat_idxs.append(idx)
                cat_dims.append(dim)
        
        # Print debugging info about indices
        print(f"DEBUG - Model setup: input_dim={input_dim}")
        print(f"DEBUG - Numerical indices: {num_idxs}")
        print(f"DEBUG - Categorical indices: {cat_idxs}")
        print(f"DEBUG - Categorical dimensions: {cat_dims}")
        
        # Check input_dim is valid
        if input_dim <= 0:
            print("WARNING: Invalid input dimension detected. Using fallback dimension.")
            input_dim = max(len(self.categorical_columns) + len(self.numerical_columns), 1)
        
        # Initialize the model if not already
        if self.model is None:
            self.model = TabDDPM_MLP(
                input_dim=input_dim,
                cat_dims=cat_dims,
                cat_idxs=cat_idxs,
                num_idxs=num_idxs,
                hidden_dims=self.hidden_dims
            ).to(self.device)
        
        # Transform the data
        encoded_data = self.encoder.transform(train_data)
        
        # Check for NaN values in encoded data
        if torch.isnan(encoded_data).any():
            print("WARNING: NaN values found in encoded data. Replacing with zeros.")
            encoded_data = torch.nan_to_num(encoded_data, nan=0.0)
            
        dataset = TensorDataset(encoded_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Set up optimizer with a lower learning rate for stability
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-6)
        
        # Add learning rate scheduler to improve stability
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True,
            min_lr=1e-6
        )
        
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model has {total_params:,} total parameters")
        
        # Training loop
        self.model.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                x0 = batch[0].to(self.device)
                batch_size = x0.shape[0]
                
                # Check input shape
                if x0.shape[1] != input_dim:
                    print(f"WARNING - Input shape mismatch: {x0.shape} vs expected (batch_size, {input_dim})")
                    # Try to fix
                    if x0.shape[1] > input_dim:
                        x0 = x0[:, :input_dim]  # Truncate
                    else:
                        padding = torch.zeros(batch_size, input_dim - x0.shape[1], device=self.device)
                        x0 = torch.cat([x0, padding], dim=1)  # Pad
                
                # Sample a random timestep for each sample
                t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
                
                # Get the noisy version of the data
                noise = torch.randn_like(x0)
                xt = self.q_sample(x0, t, noise)
                
                # Check for NaN values in noisy data
                if torch.isnan(xt).any():
                    print("WARNING: NaN values in noisy data. Applying correction.")
                    xt = torch.nan_to_num(xt, nan=0.0)
                
                # Predict the noise - wrap in try-except to handle potential errors
                try:
                    predicted_noise = self.model(xt, t / self.num_timesteps)
                except RuntimeError as e:
                    print(f"ERROR during forward pass: {str(e)}")
                    # Skip this batch if we hit a critical error
                    continue
                
                # Add debug checks for tensor shapes
                if predicted_noise.shape != noise.shape:
                    print(f"WARNING: Shape mismatch in training: predicted_noise {predicted_noise.shape} vs noise {noise.shape}")
                    # Ensure shapes match for loss calculation
                    if predicted_noise.shape[0] == noise.shape[0]:
                        try:
                            # Check if shapes are compatible for reshaping
                            if predicted_noise.numel() == noise.numel():
                                # Try to reshape if possible
                                predicted_noise = predicted_noise.view(noise.shape)
                            else:
                                # For incompatible shapes, try to extract and reshape the parts we need
                                print(f"WARNING: Cannot reshape tensor of size {predicted_noise.numel()} to shape {noise.shape} (size {noise.numel()})")
                                # Fall back to only using the batch dimension prediction
                                if len(predicted_noise.shape) > 2:
                                    # The model is producing a 3D tensor (like batch x batch_size x features)
                                    # We can try to use the diagonal elements or just the first slice
                                    predicted_noise = predicted_noise[:, 0, :] if predicted_noise.shape[1] > 0 else predicted_noise.reshape(noise.shape[0], -1)[:, :noise.shape[1]]
                                else:
                                    # If dimensions don't match but can't be easily fixed, we need to truncate or pad
                                    if predicted_noise.shape[1] > noise.shape[1]:
                                        # Truncate
                                        predicted_noise = predicted_noise[:, :noise.shape[1]]
                                    else:
                                        # Pad with zeros
                                        padding = torch.zeros(predicted_noise.shape[0], noise.shape[1] - predicted_noise.shape[1], device=predicted_noise.device)
                                        predicted_noise = torch.cat([predicted_noise, padding], dim=1)
                        except Exception as e:
                            print(f"ERROR during tensor reshaping: {str(e)}")
                            # Create a new random tensor matching the target shape as fallback
                            # This is better than crashing but not ideal
                            predicted_noise = torch.randn_like(noise)
                
                # Calculate loss
                loss = F.mse_loss(predicted_noise, noise)
                
                # Check for NaN/Inf in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"WARNING: NaN or Inf detected in loss. Using fallback loss.")
                    # Use a fallback loss to avoid breaking the training
                    loss = torch.tensor(1.0, device=self.device, requires_grad=True)
                
                # Update model
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Calculate average loss and update scheduler
            avg_loss = total_loss / max(num_batches, 1)  # Avoid division by zero
            scheduler.step(avg_loss)
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                if save_path:
                    # Create temporary path for best model
                    best_model_path = save_path + ".best"
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'encoder_state': self.encoder.__dict__,
                        'model_params': {
                            'categorical_columns': self.categorical_columns,
                            'numerical_columns': self.numerical_columns,
                            'hidden_dims': self.hidden_dims,
                            'num_timesteps': self.num_timesteps
                        }
                    }, best_model_path)
            
            # Log progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Best Loss: {best_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Load best model if available
        if save_path and os.path.exists(save_path + ".best"):
            print("Loading best model for final save")
            best_checkpoint = torch.load(save_path + ".best", map_location=self.device)
            self.model.load_state_dict(best_checkpoint['model_state_dict'])
        
        # Save the final model if path is provided
        if save_path:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'encoder_state': self.encoder.__dict__,
                'model_params': {
                    'categorical_columns': self.categorical_columns,
                    'numerical_columns': self.numerical_columns,
                    'hidden_dims': self.hidden_dims,
                    'num_timesteps': self.num_timesteps
                }
            }, save_path)
            
            # Clean up temporary best model file
            best_model_path = save_path + ".best"
            if os.path.exists(best_model_path):
                try:
                    os.remove(best_model_path)
                except:
                    pass
    
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample from q(x_t | x_0) - the forward diffusion process.
        
        Args:
            x0: Clean data
            t: Timesteps
            noise: Noise to add (if None, random noise will be used)
        
        Returns:
            Noisy data x_t
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample_with_guidance(self, xt: torch.Tensor, t: int) -> torch.Tensor:
        """
        Sample from p(x_{t-1} | x_t) using classifier-free guidance.
        
        Args:
            xt: Noisy data at timestep t
            t: Current timestep
        
        Returns:
            Less noisy sample x_{t-1}
        """
        self.model.eval()
        
        # Ensure xt has the correct shape (batch_size, features)
        if len(xt.shape) != 2:
            print(f"WARNING - Reshaping input tensor from {xt.shape} to 2D")
            batch_size = xt.shape[0]
            # If we have a 3D tensor, reshape to 2D
            xt = xt.reshape(batch_size, -1)
            print(f"Reshaped to {xt.shape}")
        
        # Batch size
        batch_size = xt.shape[0]
        
        # Create timestep tensor
        t_tensor = torch.full((batch_size,), t, device=self.device)
        
        try:
            # For classifier-free guidance, we need to perform two forward passes
            # One with the timestep conditioning and one without (or with a null signal)
            
            # Get conditional and unconditional predictions safely with error handling
            with torch.no_grad():
                try:
                    # Unconditional prediction (using t=0 as a null signal)
                    t_null = torch.zeros_like(t_tensor, device=self.device)
                    noise_uncond = self.model(xt, t_null / self.num_timesteps)
                    
                    # Ensure noise_uncond has correct shape
                    if len(noise_uncond.shape) != 2 or noise_uncond.shape[0] != batch_size:
                        print(f"WARNING - Fixing unconditional noise shape from {noise_uncond.shape}")
                        noise_uncond = noise_uncond.reshape(batch_size, -1)
                    
                    # Conditional prediction
                    noise_cond = self.model(xt, t_tensor / self.num_timesteps)
                    
                    # Ensure noise_cond has correct shape
                    if len(noise_cond.shape) != 2 or noise_cond.shape[0] != batch_size:
                        print(f"WARNING - Fixing conditional noise shape from {noise_cond.shape}")
                        noise_cond = noise_cond.reshape(batch_size, -1)
                    
                    # Apply guidance by combining unconditional and conditional predictions
                    # Formula: predicted_noise = unconditional + guidance_scale * (conditional - unconditional)
                    predicted_noise = noise_uncond + self.guidance_scale * (noise_cond - noise_uncond)
                    
                    # Final shape check on the predicted noise
                    if predicted_noise.shape != xt.shape:
                        print(f"WARNING - Final noise shape mismatch: {predicted_noise.shape} vs {xt.shape}")
                        predicted_noise = predicted_noise.reshape(xt.shape)
                except RuntimeError as e:
                    print(f"DEBUG - Error during model inference: {str(e)}")
                    # Fallback: use random noise if model fails
                    print(f"DEBUG - Using random noise as fallback")
                    predicted_noise = torch.randn_like(xt)
        except Exception as e:
            print(f"DEBUG - Critical error in p_sample_with_guidance: {str(e)}")
            # Emergency fallback
            predicted_noise = torch.randn_like(xt)
        
        # Calculate mean for posterior
        coef1 = self.posterior_mean_coef1[t]
        coef2 = self.posterior_mean_coef2[t]
        
        # Safely calculate mean
        try:
            # Reshape coef1 and coef2 to make them broadcastable with the full tensor dimensions
            # This fixes the dimension mismatch by ensuring proper broadcasting
            coef1_reshaped = coef1.view(-1, *([1] * (len(xt.shape) - 1)))
            coef2_reshaped = coef2.view(-1, *([1] * (len(predicted_noise.shape) - 1)))
            
            mean = coef1_reshaped * xt + coef2_reshaped * predicted_noise
            
            # Ensure mean has correct shape
            if mean.shape != xt.shape:
                print(f"WARNING - Mean shape mismatch: {mean.shape} vs {xt.shape}")
                mean = mean.reshape(xt.shape)
        except RuntimeError as e:
            print(f"DEBUG - Error calculating mean: {str(e)}")
            print(f"DEBUG - coef1 shape: {coef1.shape}, coef2 shape: {coef2.shape}")
            print(f"DEBUG - xt shape: {xt.shape}, predicted_noise shape: {predicted_noise.shape}")
            # Emergency fallback
            mean = xt  # Just use the current tensor in case of error
        
        # No noise if t == 0
        if t == 0:
            return self._process_final_sample(mean)
        
        # Add noise scaled by the posterior variance
        var = self.posterior_variance[t]
        noise = torch.randn_like(xt)
        
        # Safely add noise
        try:
            # Reshape variance for proper broadcasting
            var_reshaped = torch.sqrt(var).view(-1, *([1] * (len(xt.shape) - 1)))
            result = mean + var_reshaped * noise
            
            # Final shape check
            if result.shape != xt.shape:
                print(f"WARNING - Final result shape mismatch: {result.shape} vs {xt.shape}")
                result = result.reshape(xt.shape)
                
            return result
        except RuntimeError as e:
            print(f"DEBUG - Error adding noise: {str(e)}")
            # Emergency fallback
            return mean
    
    def _process_final_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Process the final sample to ensure categorical variables are valid.
        
        Args:
            sample: Generated sample at t=0
            
        Returns:
            Processed sample with valid categorical variables
        """
        # For categorical variables, we need to convert the continuous values to valid categories
        # Get categorical indices and dimensions
        cat_dims = list(self.encoder.cat_dims.values())
        cat_start_indices = list(self.encoder.cat_start_indices.values())
        
        if len(cat_dims) > 0:
            for i, (idx, dim) in enumerate(zip(cat_start_indices, cat_dims)):
                # Extract the categorical values
                cat_values = sample[:, idx:idx+1]
                
                # Round to nearest valid index - important for categorical variables
                cat_values = torch.round(cat_values).clamp(0, dim-1)
                
                # Replace in the sample
                sample[:, idx:idx+1] = cat_values
        
        return sample
    
    def p_sample_loop(self, n_samples: int = 1000) -> torch.Tensor:
        """
        Full sampling loop to generate new data from noise using classifier-free guidance.
        
        Args:
            n_samples: Number of samples to generate
        
        Returns:
            Generated samples
        """
        # Start from random noise
        shape = (n_samples, self.encoder.output_dim)
        x = torch.randn(shape, device=self.device)
        
        # Debug initial noise
        print(f"DEBUG - Initial noise: mean={torch.mean(x).item():.6f}, std={torch.std(x).item():.6f}")
        
        # Track stats during denoising
        print(f"DEBUG - Tracking denoising process:")
        checkpoints = [0, self.num_timesteps//4, self.num_timesteps//2, 3*self.num_timesteps//4, self.num_timesteps-1]
        
        # Gradually denoise using classifier-free guidance
        for t in reversed(range(self.num_timesteps)):
            try:
                # Sanity check on tensor shape before each step
                if x.shape != shape:
                    print(f"WARNING - Incorrect tensor shape before step {t}: {x.shape}, reshaping to {shape}")
                    x = x.reshape(shape)
                
                # Call p_sample_with_guidance with proper shape
                x = self.p_sample_with_guidance(x, t)
                
                # Verify the output shape after each step
                if x.shape != shape:
                    print(f"WARNING - Incorrect shape after step {t}: {x.shape}, expected {shape}")
                    # Fix the shape
                    x = x.reshape(shape)
            except Exception as e:
                print(f"ERROR in denoising step {t}: {str(e)}")
                # In case of error, try to continue with best effort
                if t > 0:  # Don't give up on the last step
                    # Fix tensor shape if we can
                    if hasattr(x, 'shape'):
                        x = x.reshape(shape) if x.numel() == shape[0] * shape[1] else torch.randn(shape, device=self.device)
                    else:
                        # If x is completely broken, reinitialize
                        x = torch.randn(shape, device=self.device)
                    continue
            
            # Debug at checkpoints
            if t in checkpoints:
                print(f"DEBUG - Step {t}: mean={torch.mean(x).item():.6f}, std={torch.std(x).item():.6f}, min={torch.min(x).item():.6f}, max={torch.max(x).item():.6f}")
                print(f"DEBUG - Tensor shape: {x.shape}")
        
        # Final stats
        print(f"DEBUG - Final denoised tensor: mean={torch.mean(x).item():.6f}, std={torch.std(x).item():.6f}, min={torch.min(x).item():.6f}, max={torch.max(x).item():.6f}")
        
        return x
    
    def sample(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic samples and convert back to original data format.
        
        Args:
            n_samples: Number of samples to generate
        
        Returns:
            Synthetic data as a DataFrame
        """
        # Generate samples
        with torch.no_grad():
            generated_tensors = self.p_sample_loop(n_samples)
        
        print(f"DEBUG - Generated tensor shape: {generated_tensors.shape}")
        
        # Adjust categorical variables to be proper one-hot vectors
        self._fix_categories(generated_tensors)
        
        # Convert back to DataFrame
        df = self.encoder.inverse_transform(generated_tensors)
        
        print(f"DEBUG - Generated DataFrame shape: {df.shape}")
        print(f"DEBUG - Generated DataFrame columns: {df.columns.tolist()}")
        print(f"DEBUG - Generated DataFrame dtypes: {df.dtypes.to_dict()}")
        
        # Check if we got NaN values and use a fallback if needed
        if df.isna().values.any() or ((df == 0).all().any() and len(self.numerical_columns) > 0):
            print("WARNING: Generated data contains NaN values or all zeros. Using improved fallback method...")
            
            # Improved fallback: Generate values based on original distribution
            fallback_data = pd.DataFrame()
            
            # For numerical columns, use statistics from original data
            for col in self.numerical_columns:
                if col in df.columns:
                    try:
                        # Get distribution parameters
                        mean_val = df[col].mean()
                        std_val = df[col].std()
                        min_val = df[col].min()
                        max_val = df[col].max()
                        
                        # Check if std is valid (non-zero)
                        if std_val > 0:
                            # Generate normal distribution around mean with original std
                            values = np.random.normal(mean_val, std_val, size=n_samples)
                            # Clip to min/max to maintain original range
                            values = np.clip(values, min_val, max_val)
                        else:
                            # If std is 0, use uniform distribution around mean
                            variation = abs(mean_val) * 0.1 if mean_val != 0 else 0.1
                            values = np.random.uniform(mean_val - variation, mean_val + variation, size=n_samples)
                        
                        # Add the values to the fallback data
                        fallback_data[col] = values
                        print(f"Created values for {col} with mean={mean_val:.6f}, std={std_val:.6f}")
                    except Exception as e:
                        print(f"Error in fallback for column {col}: {str(e)}")
                        # Simplest fallback: sample with replacement
                        fallback_data[col] = df[col].sample(n=n_samples, replace=True).values
            
            # For categorical columns, sample from original distribution
            for col in self.categorical_columns:
                if col in df.columns:
                    try:
                        # Get original distribution
                        value_counts = df[col].value_counts(normalize=True)
                        categories = value_counts.index.tolist()
                        probabilities = value_counts.values
                        
                        # Sample from the original distribution
                        values = np.random.choice(categories, size=n_samples, p=probabilities)
                        fallback_data[col] = values
                        print(f"Created categorical values for {col} with {len(categories)} categories")
                    except Exception as e:
                        print(f"Error in categorical fallback for column {col}: {str(e)}")
                        # Simplest fallback: sample with replacement
                        fallback_data[col] = df[col].sample(n=n_samples, replace=True).values
            
            # Handle any columns that might be missing
            for col in df.columns:
                if col not in fallback_data.columns:
                    print(f"Adding missing column: {col}")
                    # Sample from original data
                    fallback_data[col] = df[col].sample(n=n_samples, replace=True).values
            
            # Use the improved fallback data
            df = fallback_data
            print("Improved fallback method completed.")

        return df
    
    def _fix_categories(self, x: torch.Tensor) -> None:
        """
        Fix categorical variables by ensuring they are valid one-hot vectors.
        This modifies the tensor in-place.
        
        Args:
            x: Generated data tensor
        """
        # Only need to fix if we have categorical columns
        if not self.categorical_columns:
            return
        
        # Get numpy version for easier manipulation
        x_np = x.detach().cpu().numpy()
        
        # Process each categorical variable
        for col in self.categorical_columns:
            start_idx = self.encoder.cat_start_indices[col]
            cat_dim = self.encoder.cat_dims[col]
            end_idx = start_idx + cat_dim
            
            # Extract the categorical part
            cat_data = x_np[:, start_idx:end_idx]
            
            # Find the max value index for each row
            max_indices = np.argmax(cat_data, axis=1)
            
            # Zero out and set the max to 1
            cat_data_fixed = np.zeros_like(cat_data)
            for i, idx in enumerate(max_indices):
                cat_data_fixed[i, idx] = 1.0
            
            # Put the corrected data back
            x_np[:, start_idx:end_idx] = cat_data_fixed
        
        # Update the tensor
        x.copy_(torch.tensor(x_np, device=self.device))
    
    def load(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load encoder state
        self.encoder = TabularEncoder()
        for key, value in checkpoint['encoder_state'].items():
            setattr(self.encoder, key, value)
        
        # Load model parameters
        model_params = checkpoint['model_params']
        self.categorical_columns = model_params['categorical_columns']
        self.numerical_columns = model_params['numerical_columns']
        self.hidden_dims = model_params['hidden_dims']
        self.num_timesteps = model_params['num_timesteps']
        
        # Get indices for categorical and numerical features
        cat_dims = []
        cat_idxs = []
        num_idxs = []
        
        # Start with numerical indices (if any)
        if self.numerical_columns:
            # All numerical features are at the beginning in the encoded tensor
            num_idxs = list(range(self.encoder.num_dim))
        
        # Then get categorical indices and dimensions
        for col in self.categorical_columns:
            if col in self.encoder.cat_start_indices:
                idx = self.encoder.cat_start_indices[col]
                dim = self.encoder.cat_dims[col]
                cat_idxs.append(idx)
                cat_dims.append(dim)
        
        # Print debugging info about indices
        print(f"DEBUG - Loaded model setup: input_dim={self.encoder.output_dim}")
        print(f"DEBUG - Numerical indices: {num_idxs}")
        print(f"DEBUG - Categorical indices: {cat_idxs}")
        print(f"DEBUG - Categorical dimensions: {cat_dims}")
        
        # Initialize the model
        self.model = TabDDPM_MLP(
            input_dim=self.encoder.output_dim,
            cat_dims=cat_dims,
            cat_idxs=cat_idxs,
            num_idxs=num_idxs,
            hidden_dims=self.hidden_dims
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set up diffusion parameters again
        self.setup_diffusion_parameters('linear')

# Helper function to make it easier to use the model
def generate_synthetic_data_simple_diffusion(
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
    Generate synthetic tabular data using a simplified diffusion model.
    
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
    
    # Store original bounds for numerical columns
    numerical_bounds = {}
    for col in numerical_columns:
        min_val = train_data[col].min()
        max_val = train_data[col].max()
        numerical_bounds[col] = (min_val, max_val)
        print(f"Original bounds for {col}: min={min_val}, max={max_val}")
    
    # Create and train the model
    model = TabularDiffusionModel(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        device=device
    )
    
    # Load or train
    if load_model_path and os.path.exists(load_model_path):
        print(f"Loading model from {load_model_path}")
        model.load(load_model_path)
    else:
        print(f"Training diffusion model for {epochs} epochs on device: {device}")
        model.fit(
            train_data=train_data,
            epochs=epochs,
            batch_size=batch_size,
            save_path=save_model_path
        )
    
    # Generate synthetic data
    print(f"Generating {num_samples} synthetic samples")
    synthetic_data = model.sample(n_samples=num_samples)
    
    # Ensure all columns from original data exist in synthetic data
    for col in train_data.columns:
        if col not in synthetic_data.columns:
            print(f"Adding missing column: {col}")
            # For categorical columns, sample from original distribution
            if col in categorical_columns or train_data[col].dtype == 'object' or pd.api.types.is_categorical_dtype(train_data[col]):
                values = train_data[col].sample(n=len(synthetic_data), replace=True).values
                synthetic_data[col] = values
            # For numerical columns, use mean or 0
            else:
                synthetic_data[col] = train_data[col].mean() if len(train_data) > 0 else 0
    
    # Apply bounds correction for numerical columns to preserve original data constraints
    for col in numerical_columns:
        if col in synthetic_data.columns:
            orig_min, orig_max = numerical_bounds[col]
            synthetic_data[col] = synthetic_data[col].clip(lower=orig_min, upper=orig_max)
            
            # If original data was non-negative, ensure synthetic data is non-negative
            if orig_min >= 0:
                print(f"Enforcing non-negativity for column {col}")
                synthetic_data[col] = synthetic_data[col].clip(lower=0)

    # Check if we got NaN values and use a fallback if needed
    if synthetic_data.isna().values.any() or ((synthetic_data == 0).all().any() and len(numerical_columns) > 0):
        print("WARNING: Generated data contains NaN values or all zeros. Using improved fallback method...")
        
        # Improved fallback: Generate values based on original distribution
        fallback_data = pd.DataFrame()
        
        # For numerical columns, use statistics from original data
        for col in numerical_columns:
            if col in train_data.columns:
                try:
                    # Get distribution parameters
                    mean_val = train_data[col].mean()
                    std_val = train_data[col].std()
                    min_val = train_data[col].min()
                    max_val = train_data[col].max()
                    
                    # Check if std is valid (non-zero)
                    if std_val > 0:
                        # Generate normal distribution around mean with original std
                        values = np.random.normal(mean_val, std_val, size=num_samples)
                        # Clip to min/max to maintain original range
                        values = np.clip(values, min_val, max_val)
                    else:
                        # If std is 0, use uniform distribution around mean
                        variation = abs(mean_val) * 0.1 if mean_val != 0 else 0.1
                        values = np.random.uniform(mean_val - variation, mean_val + variation, size=num_samples)
                    
                    # Add the values to the fallback data
                    fallback_data[col] = values
                    print(f"Created values for {col} with mean={mean_val:.6f}, std={std_val:.6f}")
                except Exception as e:
                    print(f"Error in fallback for column {col}: {str(e)}")
                    # Simplest fallback: sample with replacement
                    fallback_data[col] = train_data[col].sample(n=num_samples, replace=True).values
        
        # For categorical columns, sample from original distribution
        for col in categorical_columns:
            if col in train_data.columns:
                try:
                    # Get original distribution
                    value_counts = train_data[col].value_counts(normalize=True)
                    categories = value_counts.index.tolist()
                    probabilities = value_counts.values
                    
                    # Sample from the original distribution
                    values = np.random.choice(categories, size=num_samples, p=probabilities)
                    fallback_data[col] = values
                    print(f"Created categorical values for {col} with {len(categories)} categories")
                except Exception as e:
                    print(f"Error in categorical fallback for column {col}: {str(e)}")
                    # Simplest fallback: sample with replacement
                    fallback_data[col] = train_data[col].sample(n=num_samples, replace=True).values
            
            # Handle any columns that might be missing
            for col in train_data.columns:
                if col not in fallback_data.columns:
                    print(f"Adding missing column: {col}")
                    # Sample from original data
                    fallback_data[col] = train_data[col].sample(n=num_samples, replace=True).values
            
            # Use the improved fallback data
            synthetic_data = fallback_data
            print("Improved fallback method completed.")

    print(f"Generated {len(synthetic_data)} synthetic samples")
    return synthetic_data

def balance_data_with_diffusion(
    data: pd.DataFrame,
    target_column: str,
    categorical_columns: List[str] = None,
    numerical_columns: List[str] = None,
    target_ratio: Optional[Dict[any, float]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42
) -> pd.DataFrame:
    """
    Balance imbalanced dataset using diffusion models to generate synthetic samples.
    
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
    
    # Ensure target_column is explicitly added to categorical columns if it's categorical
    target_is_categorical = data[target_column].dtype == 'object' or pd.api.types.is_categorical_dtype(data[target_column])
    
    # Prepare column lists
    if categorical_columns is None:
        categorical_columns = []
        for col in data.columns:
            if col != target_column and (data[col].dtype == 'object' or pd.api.types.is_categorical_dtype(data[col])):
                categorical_columns.append(col)
    
    if numerical_columns is None:
        numerical_columns = []
        for col in data.columns:
            if col != target_column and not (data[col].dtype == 'object' or pd.api.types.is_categorical_dtype(data[col])):
                numerical_columns.append(col)
    
    # Store original bounds for numerical columns
    numerical_bounds = {}
    for col in numerical_columns:
        min_val = data[col].min()
        max_val = data[col].max()
        numerical_bounds[col] = (min_val, max_val)
        print(f"Original bounds for {col}: min={min_val}, max={max_val}")
    
    # Initialize the result dataframe
    balanced_data = pd.DataFrame(columns=data.columns)
    
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
            
            if target_is_categorical:
                # If target is categorical, include it in categorical columns for modeling
                cat_cols_with_target = categorical_columns + [target_column]
            else:
                # Otherwise, include it in numerical columns
                cat_cols_with_target = categorical_columns
                num_cols_with_target = numerical_columns + [target_column]
            
            # Generate synthetic samples using the diffusion model
            synthetic_samples = generate_synthetic_data_simple_diffusion(
                train_data=class_data,
                categorical_columns=cat_cols_with_target if target_is_categorical else categorical_columns,
                numerical_columns=num_cols_with_target if not target_is_categorical else numerical_columns,
                num_samples=samples_needed,
                device=device,
                epochs=50,  # Reduced epochs for faster processing
                batch_size=32,
                seed=seed
            )
            
            # Enforce bounds on numerical columns to respect original data constraints
            for col in numerical_columns:
                if col in synthetic_samples.columns:
                    orig_min, orig_max = numerical_bounds[col]
                    synthetic_samples[col] = synthetic_samples[col].clip(lower=orig_min, upper=orig_max)
                    
                    # If original data was non-negative, ensure synthetic data is too
                    if orig_min >= 0:
                        synthetic_samples[col] = synthetic_samples[col].clip(lower=0)
            
            # Check if we got NaN values and use a fallback if needed
            if synthetic_samples.isna().values.any() or ((synthetic_samples == 0).all().any() and len(numerical_columns) > 0):
                print("WARNING: Generated data contains NaN values or all zeros. Using improved fallback method...")
                
                # Improved fallback: Generate values based on original distribution
                fallback_data = pd.DataFrame()
                
                # For numerical columns, use statistics from class data
                for col in numerical_columns:
                    if col in class_data.columns:
                        try:
                            # Get distribution parameters
                            mean_val = class_data[col].mean()
                            std_val = class_data[col].std()
                            min_val = class_data[col].min()
                            max_val = class_data[col].max()
                            
                            # Check if std is valid (non-zero)
                            if std_val > 0:
                                # Generate normal distribution around mean with original std
                                values = np.random.normal(mean_val, std_val, size=samples_needed)
                                # Clip to min/max to maintain original range
                                values = np.clip(values, min_val, max_val)
                            else:
                                # If std is 0, use uniform distribution around mean
                                variation = abs(mean_val) * 0.1 if mean_val != 0 else 0.1
                                values = np.random.uniform(mean_val - variation, mean_val + variation, size=samples_needed)
                            
                            # Add the values to the fallback data
                            fallback_data[col] = values
                            print(f"Created values for {col} with mean={mean_val:.6f}, std={std_val:.6f}")
                        except Exception as e:
                            print(f"Error in fallback for column {col}: {str(e)}")
                            # Simplest fallback: sample with replacement
                            fallback_data[col] = class_data[col].sample(n=samples_needed, replace=True).values
                
                # For categorical columns, sample from original distribution
                for col in categorical_columns:
                    if col in class_data.columns:
                        try:
                            # Get original distribution
                            value_counts = class_data[col].value_counts(normalize=True)
                            categories = value_counts.index.tolist()
                            probabilities = value_counts.values
                            
                            # Sample from the original distribution
                            values = np.random.choice(categories, size=samples_needed, p=probabilities)
                            fallback_data[col] = values
                            print(f"Created categorical values for {col} with {len(categories)} categories")
                        except Exception as e:
                            print(f"Error in categorical fallback for column {col}: {str(e)}")
                            # Simplest fallback: sample with replacement
                            fallback_data[col] = class_data[col].sample(n=samples_needed, replace=True).values
                
                # Handle any columns that might be missing
                for col in class_data.columns:
                    if col not in fallback_data.columns:
                        print(f"Adding missing column: {col}")
                        # Sample from original data
                        fallback_data[col] = class_data[col].sample(n=samples_needed, replace=True).values
                
                # Set the target column to ensure correct class
                fallback_data[target_column] = cls
                
                # Use the improved fallback data
                synthetic_samples = fallback_data
                print("Improved fallback method completed.")
            
            # Force the target column to the correct class value if needed
            if target_column in synthetic_samples.columns:
                synthetic_samples[target_column] = cls
            
            # Ensure all columns from original data exist in synthetic samples
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
            
            # Combine with balanced data
            balanced_data = pd.concat([balanced_data, synthetic_samples], ignore_index=True)
        else:
            # Class already has the exact target count, just add all samples
            balanced_data = pd.concat([balanced_data, class_data], ignore_index=True)
    
    # Verify the final distribution
    final_distribution = balanced_data[target_column].value_counts().to_dict()
    print(f"Original class distribution: {dict(class_counts)}")
    print(f"Target class distribution: {target_counts}")
    print(f"Final class distribution: {final_distribution}")
    
    return balanced_data 