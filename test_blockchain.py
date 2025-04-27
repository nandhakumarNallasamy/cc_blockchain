#!/usr/bin/env python3
# Tests for blockchain implementation

import unittest
import hashlib
import json
from datetime import datetime

from blockchain import Blockchain, Block

class TestBlockchain(unittest.TestCase):
    def setUp(self):
        """Set up a new blockchain for each test"""
        self.blockchain = Blockchain()
    
    def test_genesis_block(self):
        """Test that the blockchain is initialized with a genesis block"""
        self.assertEqual(len(self.blockchain.chain), 1)
        self.assertEqual(self.blockchain.chain[0].index, 0)
        self.assertEqual(self.blockchain.chain[0].previous_hash, "0")
    
    def test_add_transaction(self):
        """Test adding a transaction to pending transactions"""
        # Add a transaction
        transaction = {
            'action': 'TEST_ACTION',
            'data': 'test_data',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.blockchain.add_transaction(transaction)
        
        # Check that the transaction is in pending transactions
        self.assertEqual(len(self.blockchain.pending_transactions), 1)
        self.assertEqual(self.blockchain.pending_transactions[0]['action'], 'TEST_ACTION')
    
    def test_mine_block(self):
        """Test mining a block with pending transactions"""
        # Add some transactions
        for i in range(3):
            transaction = {
                'action': f'TEST_ACTION_{i}',
                'data': f'test_data_{i}',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.blockchain.add_transaction(transaction)
        
        # Mine a block
        block_index = self.blockchain.mine_pending_transactions("test_miner")
        
        # Check that the block was added to the chain
        self.assertEqual(len(self.blockchain.chain), 2)
        self.assertEqual(self.blockchain.chain[1].index, 1)
        self.assertEqual(len(self.blockchain.chain[1].transactions), 3)
        
        # Check that pending transactions were reset with mining reward
        self.assertEqual(len(self.blockchain.pending_transactions), 1)
        self.assertEqual(self.blockchain.pending_transactions[0]['action'], 'MINING_REWARD')
        self.assertEqual(self.blockchain.pending_transactions[0]['miner'], 'test_miner')
    
    def test_block_hashing(self):
        """Test that block hashing is correct"""
        # Create a block
        block = Block(
            index=1,
            transactions=[{'test': 'data'}],
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            previous_hash="test_hash"
        )
        
        # Compute hash
        block_string = json.dumps({
            'index': block.index,
            'transactions': block.transactions,
            'timestamp': block.timestamp,
            'previous_hash': block.previous_hash,
            'nonce': block.nonce
        }, sort_keys=True, default=str)
        expected_hash = hashlib.sha256(block_string.encode()).hexdigest()
        
        # Check that the computed hash matches the block's hash
        self.assertEqual(block.hash, expected_hash)
    
    def test_proof_of_work(self):
        """Test that proof of work produces a valid hash"""
        # Create a block
        block = Block(
            index=1,
            transactions=[{'test': 'data'}],
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            previous_hash="test_hash"
        )
        
        # Perform proof of work
        proof = self.blockchain.proof_of_work(block)
        
        # Check that the proof starts with the required number of zeros
        self.assertTrue(proof.startswith('0' * self.blockchain.difficulty))
        
        # Check that the proof is a valid hash for the block
        self.assertTrue(self.blockchain.is_valid_proof(block, proof))
    
    def test_chain_validation(self):
        """Test blockchain validation"""
        # Add some blocks
        for i in range(3):
            self.blockchain.add_transaction({
                'action': f'TEST_ACTION_{i}',
                'data': f'test_data_{i}'
            })
            self.blockchain.mine_pending_transactions(f"miner_{i}")
        
        # Verify chain is valid
        self.assertTrue(self.blockchain.is_chain_valid())
        
        # Tamper with a block and verify chain is invalid
        self.blockchain.chain[1].transactions[0]['data'] = 'tampered_data'
        self.blockchain.chain[1].hash = self.blockchain.chain[1].compute_hash()
        
        self.assertFalse(self.blockchain.is_chain_valid())
    
    def test_get_transaction_history(self):
        """Test retrieving transaction history"""
        # Add blocks with different transaction types
        for action in ['ACTION_1', 'ACTION_2', 'ACTION_1']:
            self.blockchain.add_transaction({
                'action': action,
                'data': f'data_for_{action}'
            })
            self.blockchain.mine_pending_transactions("test_miner")
        
        # Get all transactions
        all_transactions = self.blockchain.get_transaction_history()
        self.assertEqual(len(all_transactions), 3)
        
        # Get transactions filtered by type
        action_1_transactions = self.blockchain.get_transaction_history('ACTION_1')
        self.assertEqual(len(action_1_transactions), 2)
        
        action_2_transactions = self.blockchain.get_transaction_history('ACTION_2')
        self.assertEqual(len(action_2_transactions), 1)

if __name__ == '__main__':
    unittest.main()
