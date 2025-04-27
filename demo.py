#!/usr/bin/env python3
# Demo script for blockchain-based credit card fraud detection

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
import hashlib
from datetime import datetime

# Import preprocessing and model training modules
import prepare_dataset
import model_training

from blockchain import Blockchain
from model_registry import ModelRegistry

def suppress_warnings():
    """Suppress specific warnings"""
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

def load_or_prepare_data():
    """
    Load preprocessed data or prepare it if not available
    Returns processed train and test datasets
    """
    data_dir = './data'
    train_data_path = os.path.join(data_dir, 'train_data.csv')
    test_data_path = os.path.join(data_dir, 'test_data.csv')
    
    # Check if preprocessed data exists
    if os.path.exists(train_data_path) and os.path.exists(test_data_path):
        print("Loading preprocessed datasets...")
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)
    else:
        print("Preprocessed datasets not found. Preparing data...")
        # Call the prepare_dataset module to process the data
        prepare_dataset.prepare_dataset('creditcard.csv')
        
        # Reload the processed data
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)
    
    return train_df, test_df

def load_or_train_model():
    """
    Load an existing model or train a new one
    Returns trained model
    """
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Check for existing models
    model_files = [f for f in os.listdir(results_dir) if f.endswith('_model.pkl')]
    
    if model_files:
        # Load the first available model
        model_path = os.path.join(results_dir, model_files[0])
        print(f"Loading pre-trained model: {model_path}")
        return joblib.load(model_path)
    else:
        print("No pre-trained models found. Training a new model...")
        # Call the model_training module to train a model
        trained_model = model_training.main()
        return trained_model

def demo_system():
    """Run a complete demonstration of the blockchain-based fraud detection system"""
    # Suppress warnings
    suppress_warnings()

    print("\n" + "="*80)
    print("BLOCKCHAIN-BASED CREDIT CARD FRAUD DETECTION SYSTEM DEMO")
    print("="*80 + "\n")
    
    # Initialize blockchain and model registry
    print("Initializing blockchain and model registry...")
    blockchain = Blockchain()
    model_registry = ModelRegistry()
    
    # Create output directory
    os.makedirs('demo_output', exist_ok=True)
    
    # Load or prepare datasets
    train_df, test_df = load_or_prepare_data()
    
    # Load or train model
    model = load_or_train_model()
    
    # Prepare test data
    X_test = test_df.drop('Class', axis=1)
    y_test = test_df['Class']
    
    # Convert to numpy for prediction
    X_test_array = X_test.to_numpy()
    
    print("\n" + "-"*40)
    print("DATASET INFORMATION")
    print("-"*40)
    print(f"Test dataset shape: {test_df.shape}")
    print(f"Number of transactions: {len(test_df)}")
    print(f"Fraud cases: {test_df['Class'].sum()}")
    print(f"Fraud percentage: {test_df['Class'].mean() * 100:.4f}%")
    
    # Evaluation
    from sklearn.metrics import classification_report, roc_curve, auc
    
    # Predict
    y_pred = model.predict(X_test_array)
    y_pred_proba = model.predict_proba(X_test_array)[:, 1]
    
    # Print classification report
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('demo_output/roc_curve.png')
    plt.close()
    
    # Model registration and blockchain
    # Compute model hash
    model_str = str(model.get_params())
    if hasattr(model, 'feature_importances_'):
        model_str += str(model.feature_importances_.tolist())
    model_hash = hashlib.sha256(model_str.encode()).hexdigest()
    
    # Create model metadata
    model_metadata = {
        'model_hash': model_hash,
        'model_type': type(model).__name__,
        'hyperparameters': model.get_params(),
        'metrics': {
            'accuracy': (y_pred == y_test).mean(),
            'auc': roc_auc,
            'precision_fraud': classification_report(y_test, y_pred, output_dict=True)['1']['precision'],
            'recall_fraud': classification_report(y_test, y_pred, output_dict=True)['1']['recall'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Register in model registry
    model_id = model_registry.register_model(model_metadata)
    
    # Record in blockchain
    transaction_data = {
        'action': 'MODEL_REGISTRATION',
        'model_id': model_id,
        'model_hash': model_hash,
        'metrics': model_metadata['metrics'],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    blockchain.add_transaction(transaction_data)
    blockchain.mine_pending_transactions("model_developer")
    
    print(f"\nModel registered with ID: {model_id}")
    print(f"Model hash: {model_hash}")
    
    # Transaction processing simulation
    print("\n" + "-"*40)
    print("TRANSACTION PROCESSING SIMULATION")
    print("-"*40)
    
    # Sample transactions
    sample_size = 10
    sample_indices = np.random.choice(X_test_array.shape[0], sample_size, replace=False)
    sample_transactions = X_test_array[sample_indices]
    sample_labels = y_test.iloc[sample_indices]
    
    print(f"Processing {sample_size} sample transactions...")
    
    for i in range(sample_size):
        # Make prediction
        transaction = sample_transactions[i].reshape(1, -1)
        prediction = model.predict(transaction)[0]
        prediction_proba = model.predict_proba(transaction)[0][1]
        actual_label = sample_labels.iloc[i]
        
        # Hash the transaction
        transaction_hash = hashlib.sha256(str(transaction.tolist()).encode()).hexdigest()
        
        # Record to blockchain
        transaction_record = {
            'action': 'FRAUD_DETERMINATION',
            'transaction_hash': transaction_hash,
            'model_id': model_id,
            'prediction': int(prediction),
            'confidence': float(prediction_proba),
            'actual_label': int(actual_label),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        blockchain.add_transaction(transaction_record)
        
        # Mine transactions periodically
        if (i + 1) % 3 == 0 or i == sample_size - 1:
            blockchain.mine_pending_transactions("payment_processor")
        
        # Print transaction details
        result = "FRAUD" if prediction == 1 else "LEGITIMATE"
        actual = "FRAUD" if actual_label == 1 else "LEGITIMATE"
        match = "✓" if prediction == actual_label else "✗"
        
        print(f"Transaction {i+1}: Predicted: {result} (Confidence: {prediction_proba:.4f}), Actual: {actual} {match}")
    
    # Blockchain status
    print("\n" + "-"*40)
    print("BLOCKCHAIN STATUS")
    print("-"*40)
    print(f"Blockchain length: {len(blockchain.chain)} blocks")
    
    # Verify blockchain integrity
    blockchain_valid = blockchain.is_chain_valid()
    print(f"Blockchain valid: {'Yes' if blockchain_valid else 'No'}")
    
    # Save blockchain state
    blockchain_data = [block.to_dict() for block in blockchain.chain]
    with open('demo_output/blockchain_state.json', 'w') as f:
        json.dump(blockchain_data, f, indent=2, default=str)
    
    print("Blockchain state saved to demo_output/blockchain_state.json")

if __name__ == "__main__":
    demo_system()
