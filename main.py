#!/usr/bin/env python3
# Blockchain-based Credit Card Fraud Detection System
# Main application file

import os
import pandas as pd
import numpy as np
import hashlib
import json
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from blockchain import Blockchain, Block
from model_registry import ModelRegistry

def load_data(filepath):
    """Load and preprocess the credit card transaction dataset"""
    print("Loading dataset from:", filepath)
    df = pd.read_csv(filepath)
    
    # Basic dataset information
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()}")
    print(f"Fraud percentage: {df['Class'].mean() * 100:.4f}%")
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

def train_model(X_train, y_train):
    """Train a fraud detection model"""
    print("Training fraud detection model...")
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model

def compute_model_hash(model):
    """Compute a hash of the model parameters"""
    # Get model parameters as a string
    model_str = str(model.get_params())
    
    # Add feature importances
    model_str += str(model.feature_importances_.tolist())
    
    # Compute hash
    return hashlib.sha256(model_str.encode()).hexdigest()

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return performance metrics"""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Create metrics dictionary
    metrics = {
        'accuracy': report['accuracy'],
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1-score': report['1']['f1-score'],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    print(f"Model Metrics: {json.dumps(metrics, indent=2)}")
    return metrics

def register_model(model, metrics, blockchain, model_registry):
    """Register the model in the blockchain and model registry"""
    # Compute model hash
    model_hash = compute_model_hash(model)
    
    # Create model metadata
    model_metadata = {
        'model_hash': model_hash,
        'model_type': 'RandomForestClassifier',
        'hyperparameters': model.get_params(),
        'metrics': metrics,
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Add to model registry
    model_id = model_registry.register_model(model_metadata)
    
    # Record in blockchain
    transaction_data = {
        'action': 'MODEL_REGISTRATION',
        'model_id': model_id,
        'model_hash': model_hash,
        'metrics': metrics
    }
    blockchain.add_transaction(transaction_data)
    blockchain.mine_pending_transactions("model_developer")
    
    print(f"Model registered with ID: {model_id}")
    print(f"Model hash: {model_hash}")
    return model_id

def simulate_transactions(model, X_test, blockchain, model_id, num_transactions=10):
    """Simulate processing transactions and recording decisions to blockchain"""
    print(f"Simulating {num_transactions} transaction validations...")
    
    for i in range(num_transactions):
        # Select a random transaction
        idx = np.random.randint(0, X_test.shape[0])
        transaction = X_test[idx]
        
        # Make prediction
        prediction = model.predict([transaction])[0]
        prediction_proba = model.predict_proba([transaction])[0][1]
        
        # Record the decision to blockchain
        transaction_data = {
            'action': 'FRAUD_DETERMINATION',
            'transaction_id': f"tx_{int(time.time())}_{i}",
            'model_id': model_id,
            'prediction': int(prediction),
            'confidence': float(prediction_proba),
            'transaction_hash': hashlib.sha256(str(transaction).encode()).hexdigest(),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        blockchain.add_transaction(transaction_data)
        
        # Mine every few transactions
        if (i + 1) % 3 == 0 or i == num_transactions - 1:
            blockchain.mine_pending_transactions("payment_processor")
            
        print(f"Transaction {i+1}/{num_transactions} processed. Fraud: {bool(prediction)}")

def main():
    # Initialize blockchain and model registry
    blockchain = Blockchain()
    model_registry = ModelRegistry()
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_data('creditcard.csv')
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Register model on blockchain
    model_id = register_model(model, metrics, blockchain, model_registry)
    
    # Simulate transaction processing
    simulate_transactions(model, X_test, blockchain, model_id, num_transactions=10)
    
    # Print blockchain state
    print("\nFinal Blockchain State:")
    blockchain.print_chain()
    
    # Verify blockchain integrity
    print("\nVerifying blockchain integrity...")
    print(f"Blockchain integrity check: {'PASSED' if blockchain.is_chain_valid() else 'FAILED'}")
    
    # Attempt a model tampering attack and detect it
    print("\nSimulating model tampering attack...")
    simulate_model_tampering(model, model_id, model_registry, blockchain)

def simulate_model_tampering(original_model, model_id, model_registry, blockchain):
    """Simulate a model tampering attempt and detection"""
    # Create a tampered model (change a parameter)
    tampered_model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=5,  # Changed from original
        random_state=43,  # Changed from original
        class_weight='balanced'
    )
    
    # Compute hashes
    original_hash = compute_model_hash(original_model)
    tampered_hash = compute_model_hash(tampered_model)
    
    print(f"Original model hash: {original_hash}")
    print(f"Tampered model hash: {tampered_hash}")
    
    # Verify using the registry
    original_metadata = model_registry.get_model(model_id)
    
    if original_metadata['model_hash'] != tampered_hash:
        print("TAMPERING DETECTED: Model hash does not match registered hash")
        
        # Record tampering detection to blockchain
        transaction_data = {
            'action': 'TAMPERING_DETECTION',
            'model_id': model_id,
            'registered_hash': original_metadata['model_hash'],
            'detected_hash': tampered_hash,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        blockchain.add_transaction(transaction_data)
        blockchain.mine_pending_transactions("regulatory_authority")
    else:
        print("Model verification passed")

if __name__ == "__main__":
    main()
