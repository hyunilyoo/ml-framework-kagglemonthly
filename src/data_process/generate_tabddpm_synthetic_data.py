"""
Script to generate synthetic data using the TabDDPM model with proper column handling
"""
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_process.custom_oversampling import generate_synthetic_data_tabddpm

def plot_distributions(orig_df, syn_df, columns, n_cols=3):
    """Plot histograms comparing original and synthetic data distributions"""
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(columns):
        if i < len(axes):
            ax = axes[i]
            sns.histplot(orig_df[col], color='blue', alpha=0.5, label='Original', ax=ax)
            sns.histplot(syn_df[col], color='red', alpha=0.5, label='Synthetic', ax=ax)
            ax.set_title(col)
            ax.legend()
    
    plt.tight_layout()
    plt.show()

def enforce_original_bounds(orig_df, syn_df, columns):
    """Ensure synthetic data respects the bounds of the original data"""
    for col in columns:
        # Get original min and max values
        orig_min = orig_df[col].min()
        orig_max = orig_df[col].max()
        
        # Apply bounds to synthetic data
        syn_df[col] = syn_df[col].clip(lower=orig_min, upper=orig_max)
        
        # Check if original data is always positive
        if orig_min >= 0:
            # Ensure synthetic data is also non-negative
            syn_df[col] = syn_df[col].clip(lower=0)
            
    return syn_df

def enforce_original_class_distribution(orig_df, syn_df, target_column):
    """Enforce the original class distribution in the synthetic data"""
    # Get original class distribution
    orig_class_dist = orig_df[target_column].value_counts(normalize=True)
    
    # Current synthetic class distribution
    syn_class_dist = syn_df[target_column].value_counts(normalize=True)
    
    print(f"Original class distribution: {orig_class_dist.to_dict()}")
    print(f"Current synthetic class distribution: {syn_class_dist.to_dict()}")
    
    # For each class in the original data
    for class_val, orig_proportion in orig_class_dist.items():
        # Calculate how many samples we should have with this class
        target_count = int(orig_proportion * len(syn_df))
        
        # Current count of this class
        current_count = (syn_df[target_column] == class_val).sum()
        
        # If we need more of this class
        if current_count < target_count:
            # How many more we need
            needed = target_count - current_count
            print(f"Need {needed} more samples with {target_column}={class_val}")
            
            # Find samples from the other class to convert
            other_class_indices = syn_df[syn_df[target_column] != class_val].index
            
            # If we have enough of the other class to convert
            if len(other_class_indices) >= needed:
                # Pick random indices from the other class
                indices_to_change = np.random.choice(other_class_indices, size=needed, replace=False)
                
                # Change the class value
                syn_df.loc[indices_to_change, target_column] = class_val
                print(f"Changed {needed} samples to {target_column}={class_val}")
            else:
                print(f"Warning: Not enough samples to convert to {target_column}={class_val}")
    
    # Check the final distribution
    final_syn_class_dist = syn_df[target_column].value_counts(normalize=True)
    print(f"Final synthetic class distribution: {final_syn_class_dist.to_dict()}")
    
    return syn_df

def main():
    # Load your data
    df = pd.read_csv('input/month4_25/train_fillna.csv')
    # df = df.drop(['id', 'Podcast_Name', 'Episode_Title'], axis=1)
    
    # Before calling generate_synthetic_data_tabddpm
    
    # Define your target variable
    target = 'Listening_Time_minutes'
    
    # Define your numerical columns (all columns except target)
    num_list = [col for col in df.columns if col != target]
    
    print(f"Target column: {target}")
    print(f"Numerical columns: {num_list}")
    print(f"Original data shape: {df.shape}")
    print(f"Original class distribution: {df[target].value_counts().to_dict()}")
    
    # Since 'rainfall' is binary (0/1), we can treat it as categorical
    # This approach ensures exact preservation of 0/1 values
    syn_data = generate_synthetic_data_tabddpm(
        train_data=df,
        categorical_columns=[target],  # Pass target as a list with one element
        numerical_columns=num_list,    # All other columns as numerical
        num_samples=1000,
        epochs=1000,                    # More epochs but smaller steps
        batch_size=64,  # Smaller batch size
        hidden_dims=[512, 512, 512],   # Deeper model as per TabDDPM paper
        guidance_scale=3.0,            # Increased from 1.0 to improve class distribution preservation
        device='cuda'
        # device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Verify the target column is in the synthetic data
    if target not in syn_data.columns:
        print(f"Target column '{target}' is missing in synthetic data. Adding it now...")
        # Sample from original distribution for the target column
        syn_data[target] = np.random.choice(
            df[target].values, 
            size=len(syn_data), 
            p=df[target].value_counts(normalize=True).values
        )
    else:
        # Check for NaN values and handle them
        nan_count = syn_data[target].isna().sum()
        if nan_count > 0:
            print(f"Found {nan_count} NaN values in target column. Filling with values sampled from original distribution...")
            # Create a mask for NaN values
            nan_mask = syn_data[target].isna()
            
            # Get unique values and their probabilities
            unique_values = df[target].unique()
            value_counts = df[target].value_counts(normalize=True)
            probabilities = [value_counts[val] for val in unique_values]
            
            # Sample values from the original distribution for NaN positions
            sampled_values = np.random.choice(
                unique_values,
                size=nan_count,
                p=probabilities
            )
            
            # Fill NaN values with sampled values
            syn_data.loc[nan_mask, target] = sampled_values
        
        # Ensure target values are properly formatted as integers (0/1 for binary target)
        syn_data[target] = syn_data[target].round().astype(int)
    
    # Apply bounds correction to ensure synthetic data respects original data's bounds
    print("\nChecking and enforcing original data bounds...")
    syn_data = enforce_original_bounds(df, syn_data, num_list)
    
    # Fix the class distribution to match the original data
    print("\nEnforcing original class distribution for the target variable...")
    syn_data = enforce_original_class_distribution(df, syn_data, target)
    
    print(f"Synthetic data shape: {syn_data.shape}")
    print(f"Synthetic class distribution: {syn_data[target].value_counts().to_dict()}")
    
    # Verify numerical features look reasonable
    print("\nOriginal data statistics (numerical features):")
    print(df[num_list].describe())
    
    print("\nSynthetic data statistics (numerical features):")
    print(syn_data[num_list].describe())
    
    # Plot distributions for numerical columns
    print("\nPlotting distributions for comparison...")
    plot_distributions(df, syn_data, num_list)
    
    # Save synthetic data
    syn_data.to_csv('synthetic_data_tabddpm.csv', index=False)
    print("\nSynthetic data saved to 'synthetic_data_tabddpm.csv'")

if __name__ == "__main__":
    main() 