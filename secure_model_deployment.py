#!/usr/bin/env python3
# Secure model deployment using blockchain

import joblib
import json
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys

from blockchain import Blockchain
from model_registry import ModelRegistry

def load_model(model_path):
    """Load a trained model from disk"""
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def compute_model_hash(model):
    """Compute a hash of the model parameters to verify integrity"""
    # Get model parameters as a string
    model_str = str(model.get_params())
    
    # Add feature importances if available
    if hasattr(model, 'feature_importances_'):
        model_str += str(model.feature_importances_.tolist())
    
    # Compute hash
    return hashlib.sha256(model_str.encode()).hexdigest()

def register_model_on_blockchain(model, blockchain, model_registry, metrics=None):
    """Register a model in the blockchain and model registry"""
    # Compute model hash
    model_hash = compute_model_hash(model)
    
    # Create model metadata
    model_metadata = {
        'model_hash': model_hash,
        'model_type': type(model).__name__,
        'hyperparameters': model.get_params(),
        'metrics': metrics or {},
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Add to model registry
    model_id = model_registry.register_model(model_metadata)
    
    # Record in blockchain
    transaction_data = {
        'action': 'MODEL_REGISTRATION',
        'model_id': model_id,
        'model_hash': model_hash,
        'metrics': metrics or {}
    }
    blockchain.add_transaction(transaction_data)
    blockchain.mine_pending_transactions("model_developer")
    
    print(f"Model registered with ID: {model_id}")
    print(f"Model hash: {model_hash}")
    return model_id

def verify_model_integrity(model, model_id, model_registry):
    """Verify the integrity of a model by comparing hashes"""
    # Compute current model hash
    current_hash = compute_model_hash(model)
    
    # Get registered model metadata
    model_metadata = model_registry.get_model(model_id)
    
    if not model_metadata:
        print(f"Model with ID {model_id} not found in registry")
        return False
    
    registered_hash = model_metadata['model_hash']
    
    # Compare hashes
    if current_hash == registered_hash:
        print("Model integrity verified - hashes match")
        return True
    else:
        print("Model integrity check FAILED - hashes do not match")
        print(f"Registered hash: {registered_hash}")
        print(f"Current hash: {current_hash}")
        return False

def record_prediction(transaction_data, prediction, confidence, model_id, blockchain):
    """Record a prediction in the blockchain"""
    # Create transaction hash from data
    transaction_str = str(transaction_data)
    transaction_hash = hashlib.sha256(transaction_str.encode()).hexdigest()
    
    # Create blockchain record
    record = {
        'action': 'FRAUD_DETERMINATION',
        'transaction_hash': transaction_hash,
        'model_id': model_id,
        'prediction': int(prediction),
        'confidence': float(confidence),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    blockchain.add_transaction(record)
    return transaction_hash

def process_transactions(model, model_id, transactions_file, blockchain, batch_size=10):
    """Process a batch of transactions and record decisions to blockchain"""
    # Load transactions
    df = pd.read_csv(transactions_file)
    print(f"Loaded {len(df)} transactions from {transactions_file}")
    
    # Process in batches
    predictions = []
    transaction_hashes = []
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_X = batch.drop('Class', axis=1) if 'Class' in batch.columns else batch
        
        print(f"Processing batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}...")
        
        # Make predictions
        batch_preds = model.predict(batch_X)
        batch_probs = model.predict_proba(batch_X)[:, 1]
        
        # Record each prediction
        for j, (_, transaction) in enumerate(batch.iterrows()):
            transaction_hash = record_prediction(
                transaction.to_dict(), 
                batch_preds[j],
                batch_probs[j],
                model_id,
                blockchain
            )
            predictions.append(batch_preds[j])
            transaction_hashes.append(transaction_hash)
        
        # Mine this batch of transactions
        blockchain.mine_pending_transactions("payment_processor")
    
    print(f"Processed {len(df)} transactions")
    print(f"Detected {sum(predictions)} potential fraud cases ({sum(predictions)/len(predictions)*100:.2f}%)")
    
    return predictions, transaction_hashes

def verify_prediction_record(transaction_hash, blockchain):
    """Verify that a prediction is recorded correctly in the blockchain"""
    # Get all FRAUD_DETERMINATION transactions
    transactions = blockchain.get_transaction_history('FRAUD_DETERMINATION')
    
    # Find the transaction with matching hash
    matching_txs = [tx for tx in transactions if tx['transaction_hash'] == transaction_hash]
    
    if not matching_txs:
        print(f"No record found for transaction hash {transaction_hash}")
        return None
    
    # Return the transaction details
    return matching_txs[0]

def simulate_model_tampering(model, model_id, model_registry, blockchain):
    """Simulate a model tampering attempt and detection"""
    print("\nSimulating model tampering attack...")
    
    # Get original model hash
    original_hash = compute_model_hash(model)
    print(f"Original model hash: {original_hash}")
    
    # Create a tampered version of the model (modify a parameter)
    if hasattr(model, 'n_estimators'):
        model.n_estimators = model.n_estimators // 2
    elif hasattr(model, 'C'):
        model.C = model.C * 2
    
    # Compute new hash
    tampered_hash = compute_model_hash(model)
    print(f"Tampered model hash: {tampered_hash}")
    
    # Verify using the registry
    tampering_detected = not verify_model_integrity(model, model_id, model_registry)
    
    if tampering_detected:
        print("TAMPERING DETECTED: Model hash does not match registered hash")
        
        # Record tampering detection to blockchain
        transaction_data = {
            'action': 'TAMPERING_DETECTION',
            'model_id': model_id,
            'registered_hash': model_registry.get_model(model_id)['model_hash'],
            'detected_hash': tampered_hash,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        blockchain.add_transaction(transaction_data)
        blockchain.mine_pending_transactions("regulatory_authority")
    else:
        print("Model verification passed")
    
    return tampering_detected

def main():
    # Initialize blockchain and model registry
    blockchain = Blockchain()
    model_registry = ModelRegistry()
    
    # Command line arguments
    if len(sys.argv) < 2:
        print("Usage: python secure_model_deployment.py <model_path> [transactions_file]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    transactions_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Load model
    model = load_model(model_path)
    if not model:
        sys.exit(1)
    
    # Register model on blockchain
    model_id = register_model_on_blockchain(model, blockchain, model_registry)
    
    # Verify model integrity
    verify_model_integrity(model, model_id, model_registry)
    
    # Process transactions if file provided
    if transactions_file:
        predictions, tx_hashes = process_transactions(
            model, model_id, transactions_file, blockchain)
        
        # Verify a random prediction record
        if tx_hashes:
            random_tx_hash = np.random.choice(tx_hashes)
            print(f"\nVerifying random prediction record: {random_tx_hash}")
            record = verify_prediction_record(random_tx_hash, blockchain)
            if record:
                print(f"Record found in block {record['block_index']}:")
                print(json.dumps(record, indent=2))
    
    # Simulate model tampering
    simulate_model_tampering(model, model_id, model_registry, blockchain)
    
    # Print blockchain state
    print("\nFinal Blockchain State:")
    blockchain.print_chain()
    
    # Verify blockchain integrity
    print("\nVerifying blockchain integrity...")
    print(f"Blockchain integrity check: {'PASSED' if blockchain.is_chain_valid() else 'FAILED'}")

if __name__ == "__main__":
    main()
