#!/usr/bin/env python3
# Real-time transaction processor for credit card fraud detection

import pandas as pd
import numpy as np
import joblib
import hashlib
import time
import json
import os
from datetime import datetime
import sys
import threading
import queue
import logging

from blockchain import Blockchain
from model_registry import ModelRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("transaction_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TransactionProcessor:
    """Processes credit card transactions in real-time with blockchain verification"""
    
    def __init__(self, model_path, model_id=None):
        """Initialize the transaction processor"""
        self.blockchain = Blockchain()
        self.model_registry = ModelRegistry()
        
        # Load model
        self.model = self.load_model(model_path)
        
        # Register model if needed
        if model_id is None:
            self.model_id = self.register_model()
        else:
            self.model_id = model_id
            self.verify_model_integrity()
        
        # Transaction processing
        self.transaction_queue = queue.Queue()
        self.processing_active = False
        self.processor_thread = None
        self.mining_interval = 5  # Time in seconds between blockchain mining
        self.last_mine_time = time.time()
        self.pending_count = 0
        self.processed_count = 0
        self.fraud_count = 0
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
    
    def load_model(self, model_path):
        """Load the fraud detection model"""
        logger.info(f"Loading model from {model_path}")
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded: {type(model).__name__}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            sys.exit(1)
    
    def compute_model_hash(self):
        """Compute a hash of the model parameters"""
        model_str = str(self.model.get_params())
        if hasattr(self.model, 'feature_importances_'):
            model_str += str(self.model.feature_importances_.tolist())
        return hashlib.sha256(model_str.encode()).hexdigest()
    
    def register_model(self):
        """Register the model on the blockchain"""
        logger.info("Registering model on blockchain")
        
        # Compute model hash
        model_hash = self.compute_model_hash()
        
        # Create model metadata
        model_metadata = {
            'model_hash': model_hash,
            'model_type': type(self.model).__name__,
            'hyperparameters': self.model.get_params(),
            'registered_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Register in model registry
        model_id = self.model_registry.register_model(model_metadata)
        
        # Record in blockchain
        transaction_data = {
            'action': 'MODEL_REGISTRATION',
            'model_id': model_id,
            'model_hash': model_hash,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.blockchain.add_transaction(transaction_data)
        self.blockchain.mine_pending_transactions("model_developer")
        
        logger.info(f"Model registered with ID: {model_id}")
        return model_id
    
    def verify_model_integrity(self):
        """Verify the integrity of the loaded model"""
        logger.info(f"Verifying model integrity for model ID: {self.model_id}")
        
        # Compute current model hash
        current_hash = self.compute_model_hash()
        
        # Get registered model metadata
        model_metadata = self.model_registry.get_model(self.model_id)
        
        if not model_metadata:
            logger.error(f"Model with ID {self.model_id} not found in registry")
            return False
        
        registered_hash = model_metadata['model_hash']
        
        # Compare hashes
        if current_hash == registered_hash:
            logger.info("Model integrity verified - hashes match")
            return True
        else:
            logger.error("MODEL INTEGRITY CHECK FAILED - HASHES DO NOT MATCH")
            logger.error(f"Registered hash: {registered_hash}")
            logger.error(f"Current hash: {current_hash}")
            
            # Record tampering detection in blockchain
            transaction_data = {
                'action': 'TAMPERING_DETECTION',
                'model_id': self.model_id,
                'registered_hash': registered_hash,
                'detected_hash': current_hash,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.blockchain.add_transaction(transaction_data)
            self.blockchain.mine_pending_transactions("regulatory_authority")
            
            return False
    
    def process_transaction(self, transaction):
        """Process a single transaction and detect potential fraud"""
        # Convert transaction to numpy array
        if isinstance(transaction, dict):
            # Extract features from dictionary
            features = []
            for feature in sorted(transaction.keys()):
                if feature != 'Class':  # Exclude target if present
                    features.append(float(transaction[feature]))
            transaction_array = np.array(features).reshape(1, -1)
        elif isinstance(transaction, pd.Series):
            # Convert pandas Series to numpy array
            transaction_array = transaction.drop('Class', errors='ignore').values.reshape(1, -1)
        else:
            # Assume it's already a numpy array
            transaction_array = transaction.reshape(1, -1)
        
        # Make prediction
        prediction = int(self.model.predict(transaction_array)[0])
        prediction_proba = float(self.model.predict_proba(transaction_array)[0][1])
        
        # Create transaction hash
        transaction_str = str(transaction)
        transaction_hash = hashlib.sha256(transaction_str.encode()).hexdigest()
        
        # Create blockchain record
        record = {
            'action': 'FRAUD_DETERMINATION',
            'transaction_hash': transaction_hash,
            'model_id': self.model_id,
            'prediction': prediction,
            'confidence': prediction_proba,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add to blockchain
        self.blockchain.add_transaction(record)
        self.pending_count += 1
        
        # Mine if needed
        current_time = time.time()
        if current_time - self.last_mine_time >= self.mining_interval:
            self.blockchain.mine_pending_transactions("payment_processor")
            self.last_mine_time = current_time
            self.pending_count = 0
        
        # Update statistics
        self.processed_count += 1
        if prediction == 1:
            self.fraud_count += 1
        
        return {
            'transaction_hash': transaction_hash,
            'prediction': prediction,
            'probability': prediction_proba,
            'is_fraud': bool(prediction)
        }
    
    def process_batch(self, transactions):
        """Process a batch of transactions"""
        results = []
        for tx in transactions:
            result = self.process_transaction(tx)
            results.append(result)
        
        # Ensure mining after batch
        if self.pending_count > 0:
            self.blockchain.mine_pending_transactions("payment_processor")
            self.last_mine_time = time.time()
            self.pending_count = 0
        
        return results
    
    def start_processing(self):
        """Start the transaction processing thread"""
        if self.processing_active:
            logger.warning("Processing thread already running")
            return
        
        self.processing_active = True
        self.processor_thread = threading.Thread(target=self._processing_loop)
        self.processor_thread.daemon = True
        self.processor_thread.start()
        logger.info("Transaction processing started")
    
    def stop_processing(self):
        """Stop the transaction processing thread"""
        if not self.processing_active:
            logger.warning("Processing thread not running")
            return
        
        self.processing_active = False
        if self.processor_thread:
            self.processor_thread.join(timeout=5.0)
        logger.info("Transaction processing stopped")
        
        # Ensure all pending transactions are mined
        if self.pending_count > 0:
            self.blockchain.mine_pending_transactions("payment_processor")
            self.pending_count = 0
    
    def _processing_loop(self):
        """Main processing loop that runs in a separate thread"""
        while self.processing_active:
            try:
                # Get transaction from queue with timeout
                try:
                    transaction = self.transaction_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process transaction
                result = self.process_transaction(transaction)
                
                # Log result
                is_fraud = "FRAUD" if result['is_fraud'] else "LEGITIMATE"
                logger.info(f"Transaction {result['transaction_hash'][:8]}: {is_fraud} (Confidence: {result['probability']:.4f})")
                
                # Mark task as done
                self.transaction_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
    
    def add_transaction(self, transaction):
        """Add a transaction to the processing queue"""
        self.transaction_queue.put(transaction)
    
    def get_blockchain_status(self):
        """Get the current status of the blockchain"""
        chain_length = len(self.blockchain.chain)
        is_valid = self.blockchain.is_chain_valid()
        
        # Get transaction counts by type
        transactions = self.blockchain.get_transaction_history()
        transaction_types = {}
        for tx in transactions:
            if 'action' in tx:
                action = tx['action']
                transaction_types[action] = transaction_types.get(action, 0) + 1
        
        return {
            'chain_length': chain_length,
            'is_valid': is_valid,
            'transaction_counts': transaction_types,
            'processed_transactions': self.processed_count,
            'fraud_detected': self.fraud_count,
            'fraud_percentage': (self.fraud_count / self.processed_count * 100) if self.processed_count > 0 else 0
        }
    
    def save_blockchain_state(self):
        """Save the current blockchain state to a file"""
        blockchain_data = []
        for block in self.blockchain.chain:
            blockchain_data.append(block.to_dict())
        
        filename = f"results/blockchain_state_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(blockchain_data, f, indent=2, default=str)
        
        logger.info(f"Blockchain state saved to {filename}")
        return filename

def process_csv_file(processor, csv_file, batch_size=100):
    """Process transactions from a CSV file"""
    logger.info(f"Processing transactions from {csv_file}")
    
    # Load CSV file
    df = pd.read_csv(csv_file)
    total_rows = len(df)
    logger.info(f"Loaded {total_rows} transactions")
    
    # Process in batches
    results = []
    for i in range(0, total_rows, batch_size):
        batch = df.iloc[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(total_rows-1)//batch_size + 1}...")
        
        # Process batch
        batch_results = processor.process_batch(batch.iterrows())
        results.extend(batch_results)
    
    # Summarize results
    fraud_count = sum(1 for r in results if r['is_fraud'])
    logger.info(f"Processed {total_rows} transactions")
    logger.info(f"Detected {fraud_count} potential fraud cases ({fraud_count/total_rows*100:.2f}%)")
    
    return results

def main():
    """Main function to run the transaction processor"""
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python transaction_processor.py <model_path> [csv_file]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    csv_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Initialize transaction processor
    processor = TransactionProcessor(model_path)
    
    if csv_file:
        # Process transactions from file
        results = process_csv_file(processor, csv_file)
        
        # Save results
        with open("results/processing_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save blockchain state
        processor.save_blockchain_state()
    else:
        # Start interactive processing
        processor.start_processing()
        
        try:
            while True:
                # Print status periodically
                status = processor.get_blockchain_status()
                print("\nCurrent Status:")
                print(f"Blockchain length: {status['chain_length']} blocks")
                print(f"Blockchain valid: {'Yes' if status['is_valid'] else 'No'}")
                print(f"Processed transactions: {status['processed_transactions']}")
                print(f"Fraud detected: {status['fraud_detected']} ({status['fraud_percentage']:.2f}%)")
                
                # Sleep for a bit
                time.sleep(10)
                
        except KeyboardInterrupt:
            print("\nStopping processing...")
            processor.stop_processing()
            
            # Save final state
            processor.save_blockchain_state()
            print("Processing completed.")

if __name__ == "__main__":
    main()
