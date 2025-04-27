#!/usr/bin/env python3
# Blockchain implementation for securing fraud detection models

import hashlib
import json
import time
from datetime import datetime

class Block:
    def __init__(self, index, transactions, timestamp, previous_hash, nonce=0):
        """Initialize a new block in the blockchain"""
        self.index = index
        self.transactions = transactions
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.compute_hash()
    
    def compute_hash(self):
        """Compute SHA-256 hash of the block"""
        block_string = json.dumps(self.__dict__, sort_keys=True, default=str)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def to_dict(self):
        """Convert block to dictionary"""
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'transactions': self.transactions,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'hash': self.hash
        }

class Blockchain:
    # Difficulty for proof of work (number of leading zeros in hash)
    difficulty = 2
    
    def __init__(self):
        """Initialize a new blockchain"""
        self.chain = []
        self.pending_transactions = []
        # Create genesis block
        self.create_genesis_block()
    
    def create_genesis_block(self):
        """Create the first block in the chain (genesis block)"""
        genesis_block = Block(0, [], datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "0")
        genesis_block.hash = genesis_block.compute_hash()
        self.chain.append(genesis_block)
    
    @property
    def last_block(self):
        """Return the last block in the chain"""
        return self.chain[-1]
    
    def proof_of_work(self, block):
        """Perform proof of work to find a valid hash"""
        block.nonce = 0
        computed_hash = block.compute_hash()
        
        while not computed_hash.startswith('0' * self.difficulty):
            block.nonce += 1
            computed_hash = block.compute_hash()
        
        return computed_hash
    
    def add_block(self, block, proof):
        """Add a new block to the chain after verification"""
        previous_hash = self.last_block.hash
        
        # Check if previous_hash in block matches the hash of the last block
        if previous_hash != block.previous_hash:
            return False
        
        # Check if proof is valid
        if not self.is_valid_proof(block, proof):
            return False
        
        block.hash = proof
        self.chain.append(block)
        return True
    
    def is_valid_proof(self, block, block_hash):
        """Check if block_hash is valid proof of work"""
        return (block_hash.startswith('0' * self.difficulty) and
                block_hash == block.compute_hash())
    
    def add_transaction(self, transaction):
        """Add a new transaction to pending transactions"""
        self.pending_transactions.append(transaction)
    
    def mine_pending_transactions(self, miner_address):
        """Mine pending transactions and add them to the blockchain"""
        if not self.pending_transactions:
            return False
        
        last_block = self.last_block
        
        # Create new block
        new_block = Block(
            index=last_block.index + 1,
            transactions=self.pending_transactions,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            previous_hash=last_block.hash
        )
        
        # Find valid proof of work
        proof = self.proof_of_work(new_block)
        
        # Add block to chain
        self.add_block(new_block, proof)
        
        # Reset pending transactions and add mining reward
        self.pending_transactions = [{
            'action': 'MINING_REWARD',
            'miner': miner_address,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }]
        
        return new_block.index
    
    def is_chain_valid(self):
        """Verify the integrity of the blockchain"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            # Check if current block hash is valid
            if current_block.hash != current_block.compute_hash():
                print(f"Invalid hash in block {i}")
                return False
            
            # Check if current block points to previous block's hash
            if current_block.previous_hash != previous_block.hash:
                print(f"Invalid chain linkage in block {i}")
                return False
        
        return True
    
    def print_chain(self):
        """Print the current state of the blockchain"""
        for i, block in enumerate(self.chain):
            print(f"Block {i}:")
            print(f"  Timestamp: {block.timestamp}")
            print(f"  Transactions: {len(block.transactions)}")
            print(f"  Previous Hash: {block.previous_hash}")
            print(f"  Hash: {block.hash}")
            print()
    
    def get_transaction_history(self, filter_type=None):
        """Get all transactions in the blockchain, optionally filtered by type"""
        transactions = []
        
        for block in self.chain:
            for tx in block.transactions:
                if filter_type is None or (
                    'action' in tx and tx['action'] == filter_type
                ):
                    tx_with_block = tx.copy()
                    tx_with_block['block_index'] = block.index
                    tx_with_block['block_hash'] = block.hash
                    transactions.append(tx_with_block)
        
        return transactions
