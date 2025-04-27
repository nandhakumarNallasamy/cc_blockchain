#!/usr/bin/env python3
# Demo script for blockchain-based credit card fraud detection

import pandas as pd
import numpy as np
import joblib
import json
import hashlib
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns

from blockchain import Blockchain
from model_registry import ModelRegistry

def demo_system():
    """Run a complete demonstration of the blockchain-based fraud detection system"""
    print("\n" + "="*80)
    print("BLOCKCHAIN-BASED CREDIT CARD FRAUD DETECTION SYSTEM DEMO")
    print("="*80 + "\n")
    
    # Initialize blockchain and model registry
    print("Initializing blockchain and model registry...")
    blockchain = Blockchain()
    model_registry = ModelRegistry()
    
    # Check if dataset exists, if not, suggest downloading
    if not os.path.exists('creditcard.csv'):
        print("ERROR: Dataset 'creditcard.csv' not found.")
        print("Please download the Credit Card Fraud Detection dataset from Kaggle:")
        print("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        return
    
    # Step 1: Load and explore dataset
    print("\n" + "-"*40)
    print("STEP 1: LOADING AND EXPLORING DATASET")
    print("-"*40)
    
    print("Loading credit card transaction dataset...")
    df = pd.read_csv('creditcard.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Number of transactions: {len(df)}")
    print(f"Fraud cases: {df['Class'].sum()}")
    print(f"Fraud percentage: {df['Class'].mean() * 100:.4f}%")
    
    # Create a directory for outputs
    os.makedirs('demo_output', exist_ok=True)
    
    # Plot class distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Class', data=df)
    plt.title('Class Distribution (0: Legitimate, 1: Fraud)')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('demo_output/class_distribution.png')
    
    # Step 2: Prepare data for model training
    print("\n" + "-"*40)
    print("STEP 2: PREPARING DATA FOR MODEL TRAINING")
    print("-"*40)
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Sample a smaller subset for this demo
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training set fraud cases: {y_train.sum()} ({y_train.mean()*100:.4f}%)")
    
    # Step 3: Train a fraud detection model
    print("\n" + "-"*40)
    print("STEP 3: TRAINING FRAUD DETECTION MODEL")
    print("-"*40)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("Training a Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    print("\nModel evaluation:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    model_path = 'demo_output/fraud_detection_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Step 4: Register model on blockchain
    print("\n" + "-"*40)
    print("STEP 4: REGISTERING MODEL ON BLOCKCHAIN")
    print("-"*40)
    
    # Compute model hash
    model_str = str(model.get_params())
    model_str += str(model.feature_importances_.tolist())
    model_hash = hashlib.sha256(model_str.encode()).hexdigest()
    
    # Create model metadata
    model_metadata = {
        'model_hash': model_hash,
        'model_type': 'RandomForestClassifier',
        'hyperparameters': model.get_params(),
        'metrics': {
            'accuracy': (y_pred == y_test).mean(),
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
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    blockchain.add_transaction(transaction_data)
    blockchain.mine_pending_transactions("model_developer")
    
    print(f"Model registered with ID: {model_id}")
    print(f"Model hash: {model_hash}")
    print("Model registration recorded in blockchain")
    
    # Step 5: Process transactions and record decisions
    print("\n" + "-"*40)
    print("STEP 5: PROCESSING TRANSACTIONS")
    print("-"*40)
    
    # Select a small sample of transactions for demonstration
    sample_size = 10
    test_sample = X_test.sample(sample_size, random_state=42)
    test_sample_indices = test_sample.index
    sample_y = y_test[test_sample_indices]
    
    print(f"Processing {sample_size} sample transactions...")
    
    for i, (idx, transaction) in enumerate(test_sample.iterrows()):
        # Make prediction
        transaction_array = transaction.values.reshape(1, -1)
        prediction = model.predict(transaction_array)[0]
        prediction_proba = model.predict_proba(transaction_array)[0][1]
        
        # Get actual class
        actual = sample_y[idx]
        
        # Create transaction hash
        transaction_str = str(transaction.to_dict())
        transaction_hash = hashlib.sha256(transaction_str.encode()).hexdigest()
        
        # Record to blockchain
        transaction_data = {
            'action': 'FRAUD_DETERMINATION',
            'transaction_id': f"tx_{i}",
            'transaction_hash': transaction_hash,
            'model_id': model_id,
            'prediction': int(prediction),
            'confidence': float(prediction_proba),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        blockchain.add_transaction(transaction_data)
        
        # Mine every few transactions
        if (i + 1) % 3 == 0 or i == sample_size - 1:
            blockchain.mine_pending_transactions("payment_processor")
        
        result = "FRAUD" if prediction == 1 else "LEGITIMATE"
        actual_result = "FRAUD" if actual == 1 else "LEGITIMATE"
        match = "✓" if prediction == actual else "✗"
        
        print(f"Transaction {i+1}: Predicted: {result} (Confidence: {prediction_proba:.4f}), Actual: {actual_result} {match}")
    
    # Step 6: Verify model integrity
    print("\n" + "-"*40)
    print("STEP 6: VERIFYING MODEL INTEGRITY")
    print("-"*40)
    
    # Get model from registry
    registered_model = model_registry.get_model(model_id)
    registered_hash = registered_model['model_hash']
    
    # Compute current hash
    current_hash = hashlib.sha256(model_str.encode()).hexdigest()
    
    print(f"Registered hash: {registered_hash}")
    print(f"Current hash:    {current_hash}")
    
    if current_hash == registered_hash:
        print("✅ Model integrity verified - hashes match")
    else:
        print("❌ Model integrity check FAILED - hashes do not match")
    
    # Step 7: Simulate tampering attempt
    print("\n" + "-"*40)
    print("STEP 7: SIMULATING TAMPERING ATTEMPT")
    print("-"*40)
    
    print("Tampering with model by modifying parameters...")
    
    # Tamper with the model by changing parameters
    tampered_model = RandomForestClassifier(
        n_estimators=50,  # Changed from 100
        max_depth=5,      # Changed from 10
        random_state=43,  # Changed from 42
        class_weight='balanced'
    )
    tampered_model.fit(X_train, y_train)
    
    # Save tampered model
    joblib.dump(tampered_model, 'demo_output/tampered_model.pkl')
    
    # Compute hash of tampered model
    tampered_str = str(tampered_model.get_params())
    tampered_str += str(tampered_model.feature_importances_.tolist())
    tampered_hash = hashlib.sha256(tampered_str.encode()).hexdigest()
    
    print(f"Original model hash: {model_hash}")
    print(f"Tampered model hash: {tampered_hash}")
    
    # Check if tampering is detected
    if tampered_hash != registered_hash:
        print("🚨 TAMPERING DETECTED: Model hash does not match registered hash")
        
        # Record tampering detection in blockchain
        tampering_data = {
            'action': 'TAMPERING_DETECTION',
            'model_id': model_id,
            'registered_hash': registered_hash,
            'detected_hash': tampered_hash,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        blockchain.add_transaction(tampering_data)
        blockchain.mine_pending_transactions("regulatory_authority")
        print("Tampering event recorded in blockchain")
    else:
        print("Model verification passed")
    
    # Step 8: View blockchain status
    print("\n" + "-"*40)
    print("STEP 8: BLOCKCHAIN STATUS")
    print("-"*40)
    
    print(f"Blockchain length: {len(blockchain.chain)} blocks")
    print(f"Blockchain valid: {'Yes' if blockchain.is_chain_valid() else 'No'}")
    
    # Count transactions by type
    transactions = blockchain.get_transaction_history()
    transaction_types = {}
    for tx in transactions:
        if 'action' in tx:
            action = tx['action']
            transaction_types[action] = transaction_types.get(action, 0) + 1
    
    print("\nTransaction types:")
    for tx_type, count in transaction_types.items():
        print(f"  {tx_type}: {count}")
    
    # Save blockchain state to file
    blockchain_data = []
    for block in blockchain.chain:
        blockchain_data.append(block.to_dict())
    
    with open('demo_output/blockchain_state.json', 'w') as f:
        json.dump(blockchain_data, f, indent=2, default=str)
    
    print("\nBlockchain state saved to demo_output/blockchain_state.json")
    
    # Save model registry to file
    with open('demo_output/model_registry.json', 'w') as f:
        json.dump(model_registry.registry, f, indent=2, default=str)
    
    print("Model registry saved to demo_output/model_registry.json")
    
    print("\n" + "="*80)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nOutput files can be found in the 'demo_output' directory.")

if __name__ == "__main__":
    demo_system()
