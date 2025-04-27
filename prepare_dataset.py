#!/usr/bin/env python3
# Prepare the credit card fraud detection dataset for model training

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

def prepare_dataset(filepath, output_dir='./data'):
    """
    Load, explore, and preprocess the credit card fraud detection dataset.
    Save the preprocessed data to disk.
    """
    print("Loading dataset...")
    df = pd.read_csv("creditcard.csv")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Basic dataset information
    print(f"Dataset shape: {df.shape}")
    print(f"Number of transactions: {len(df)}")
    print(f"Number of features: {len(df.columns) - 1}")  # Exclude 'Class'
    print(f"Fraud cases: {df['Class'].sum()}")
    print(f"Legitimate cases: {len(df) - df['Class'].sum()}")
    print(f"Fraud percentage: {df['Class'].mean() * 100:.4f}%")
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    print(f"Missing values: {missing_values}")
    
    # Analyze class distribution
    print("\nClass distribution:")
    class_dist = df['Class'].value_counts().to_dict()
    for label, count in class_dist.items():
        print(f"  Class {label}: {count} ({count/len(df)*100:.4f}%)")
    
    # Plot class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Class', data=df)
    plt.title('Class Distribution (0: Legitimate, 1: Fraud)')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    
    # Analyze transaction amounts
    print("\nTransaction amount statistics:")
    print(df['Amount'].describe())
    
    # Plot amount distribution for fraud vs. legitimate
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Amount', hue='Class', bins=50, kde=True)
    plt.title('Transaction Amount Distribution by Class')
    plt.xlabel('Amount')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'amount_distribution.png'))
    
    # Scale the 'Amount' and 'Time' features
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Time_scaled'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
    # Create a preprocessed version of the dataset
    preprocessed_df = df.copy()
    
    # Split the data
    X = preprocessed_df.drop('Class', axis=1)
    y = preprocessed_df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\nData splitting:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training set fraud cases: {y_train.sum()} ({y_train.mean()*100:.4f}%)")
    print(f"Test set fraud cases: {y_test.sum()} ({y_test.mean()*100:.4f}%)")
    
    # Save the preprocessed data
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)
    
    print(f"\nPreprocessed data saved to {output_dir}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler
    }

if __name__ == "__main__":
    prepare_dataset('creditcard.csv')
