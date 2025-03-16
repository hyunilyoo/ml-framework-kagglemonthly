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

def main():
    # Load your data
    df = pd.read_csv('input/month3/train.csv')
    df = df.drop(['id', 'day'], axis=1)
    
    # Define your target variable
    target = 'rainfall'
    
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
        epochs=500,                    # Reduced for faster testing but still meaningful
        batch_size=128, 
        hidden_dims=[512, 512, 512],   # Deeper model as per TabDDPM paper
        guidance_scale=3.0,            # Classifier-free guidance scale
        device='cuda' if torch.cuda.is_available() else 'cpu'
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