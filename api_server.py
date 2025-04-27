#!/usr/bin/env python3
# API server for blockchain-based credit card fraud detection

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import json
import joblib
import hashlib
import os
from datetime import datetime

from blockchain import Blockchain
from model_registry import ModelRegistry

app = Flask(__name__)

# Initialize blockchain and model registry
blockchain = Blockchain()
model_registry = ModelRegistry()

# Global variables for model and model_id
current_model = None
current_model_id = None
scaler = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'OK', 'blockchain_length': len(blockchain.chain)})

@app.route('/load_model', methods=['POST'])
def load_model():
    """Load a model from disk and register it on the blockchain"""
    global current_model, current_model_id, scaler
    
    data = request.json
    model_path = data.get('model_path')
    scaler_path = data.get('scaler_path')
    
    if not model_path:
        return jsonify({'error': 'Model path is required'}), 400
    
    try:
        # Load model
        model = joblib.load(model_path)
        
        # Load scaler if provided
        if scaler_path and os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        
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
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Register on model registry
        model_id = model_registry.register_model(model_metadata)
        
        # Record in blockchain
        transaction_data = {
            'action': 'MODEL_REGISTRATION',
            'model_id': model_id,
            'model_hash': model_hash
        }
        blockchain.add_transaction(transaction_data)
        blockchain.mine_pending_transactions("model_developer")
        
        # Set global variables
        current_model = model
        current_model_id = model_id
        
        return jsonify({
            'status': 'success',
            'model_id': model_id,
            'model_hash': model_hash,
            'model_type': type(model).__name__
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make a fraud prediction using the current model"""
    global current_model, current_model_id
    
    if current_model is None:
        return jsonify({'error': 'No model loaded. Call /load_model first.'}), 400
    
    try:
        # Get transaction data
        data = request.json
        transaction = data.get('transaction')
        
        if not transaction:
            return jsonify({'error': 'Transaction data is required'}), 400
        
        # Convert to numpy array
        transaction_features = []
        for feature in sorted(transaction.keys()):
            if feature != 'Class':  # Exclude target if present
                transaction_features.append(float(transaction[feature]))
        
        # Apply scaling if available
        transaction_array = np.array(transaction_features).reshape(1, -1)
        if scaler is not None:
            transaction_array = scaler.transform(transaction_array)
        
        # Make prediction
        prediction = int(current_model.predict(transaction_array)[0])
        prediction_proba = float(current_model.predict_proba(transaction_array)[0][1])
        
        # Create transaction hash
        transaction_str = str(transaction)
        transaction_hash = hashlib.sha256(transaction_str.encode()).hexdigest()
        
        # Record to blockchain
        blockchain_record = {
            'action': 'FRAUD_DETERMINATION',
            'transaction_hash': transaction_hash,
            'model_id': current_model_id,
            'prediction': prediction,
            'confidence': prediction_proba,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        blockchain.add_transaction(blockchain_record)
        # Mine after each prediction for demonstration purposes
        # In production, would batch mine for efficiency
        block_index = blockchain.mine_pending_transactions("payment_processor")
        
        return jsonify({
            'transaction_hash': transaction_hash,
            'prediction': prediction,
            'probability': prediction_proba,
            'is_fraud': bool(prediction),
            'blockchain_record': {
                'block_index': block_index,
                'transaction_hash': transaction_hash
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/verify_model', methods=['GET'])
def verify_model():
    """Verify the integrity of the currently loaded model"""
    global current_model, current_model_id
    
    if current_model is None or current_model_id is None:
        return jsonify({'error': 'No model loaded. Call /load_model first.'}), 400
    
    try:
        # Compute current model hash
        model_str = str(current_model.get_params())
        if hasattr(current_model, 'feature_importances_'):
            model_str += str(current_model.feature_importances_.tolist())
        current_hash = hashlib.sha256(model_str.encode()).hexdigest()
        
        # Get registered model metadata
        model_metadata = model_registry.get_model(current_model_id)
        
        if not model_metadata:
            return jsonify({
                'verified': False,
                'error': f'Model with ID {current_model_id} not found in registry'
            }), 404
        
        registered_hash = model_metadata['model_hash']
        
        # Compare hashes
        is_verified = current_hash == registered_hash
        
        # If tampering detected, record to blockchain
        if not is_verified:
            transaction_data = {
                'action': 'TAMPERING_DETECTION',
                'model_id': current_model_id,
                'registered_hash': registered_hash,
                'detected_hash': current_hash,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            blockchain.add_transaction(transaction_data)
            blockchain.mine_pending_transactions("regulatory_authority")
        
        return jsonify({
            'verified': is_verified,
            'model_id': current_model_id,
            'registered_hash': registered_hash,
            'current_hash': current_hash
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/blockchain/status', methods=['GET'])
def blockchain_status():
    """Get the current status of the blockchain"""
    try:
        # Get basic blockchain info
        chain_length = len(blockchain.chain)
        is_valid = blockchain.is_chain_valid()
        
        # Get transaction counts by type
        transactions = blockchain.get_transaction_history()
        transaction_types = {}
        for tx in transactions:
            if 'action' in tx:
                action = tx['action']
                transaction_types[action] = transaction_types.get(action, 0) + 1
        
        # Get the latest few blocks
        latest_blocks = []
        for i in range(max(0, chain_length - 5), chain_length):
            block = blockchain.chain[i]
            latest_blocks.append({
                'index': block.index,
                'timestamp': block.timestamp,
                'hash': block.hash,
                'previous_hash': block.previous_hash,
                'transaction_count': len(block.transactions)
            })
        
        return jsonify({
            'chain_length': chain_length,
            'is_valid': is_valid,
            'transaction_counts': transaction_types,
            'latest_blocks': latest_blocks
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/transaction/verify', methods=['GET'])
def verify_transaction():
    """Verify a transaction record in the blockchain"""
    tx_hash = request.args.get('hash')
    
    if not tx_hash:
        return jsonify({'error': 'Transaction hash is required'}), 400
    
    try:
        # Get fraud determination transactions
        transactions = blockchain.get_transaction_history('FRAUD_DETERMINATION')
        
        # Find matching transaction
        matching_tx = None
        for tx in transactions:
            if tx.get('transaction_hash') == tx_hash:
                matching_tx = tx
                break
        
        if matching_tx:
            return jsonify({
                'found': True,
                'transaction': matching_tx
            })
        else:
            return jsonify({
                'found': False,
                'message': f'No transaction found with hash {tx_hash}'
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
