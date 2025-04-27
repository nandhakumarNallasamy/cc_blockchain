#!/usr/bin/env python3
# Tests for model registry implementation

import unittest
import os
import json
from datetime import datetime

from model_registry import ModelRegistry

class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        """Set up a new model registry for each test with a temporary file"""
        self.test_registry_file = "test_registry.json"
        self.model_registry = ModelRegistry(self.test_registry_file)
    
    def tearDown(self):
        """Clean up after each test"""
        if os.path.exists(self.test_registry_file):
            os.remove(self.test_registry_file)
    
    def test_register_model(self):
        """Test registering a model in the registry"""
        # Create model metadata
        model_metadata = {
            'model_hash': 'test_hash',
            'model_type': 'TestModel',
            'hyperparameters': {'param1': 'value1', 'param2': 'value2'},
            'metrics': {'accuracy': 0.95, 'f1': 0.94}
        }
        
        # Register model
        model_id = self.model_registry.register_model(model_metadata)
        
        # Check that the model was registered
        self.assertIn(model_id, self.model_registry.registry)
        self.assertEqual(self.model_registry.registry[model_id]['model_hash'], 'test_hash')
        self.assertEqual(self.model_registry.registry[model_id]['model_type'], 'TestModel')
        
        # Check that the file was created
        self.assertTrue(os.path.exists(self.test_registry_file))
    
    def test_get_model(self):
        """Test retrieving a model from the registry"""
        # Register a model
        model_metadata = {
            'model_hash': 'test_hash',
            'model_type': 'TestModel'
        }
        model_id = self.model_registry.register_model(model_metadata)
        
        # Get the model
        retrieved_model = self.model_registry.get_model(model_id)
        
        # Check that the retrieved model is correct
        self.assertEqual(retrieved_model['model_hash'], 'test_hash')
        self.assertEqual(retrieved_model['model_type'], 'TestModel')
        
        # Try to get a non-existent model
        non_existent = self.model_registry.get_model('non_existent_id')
        self.assertIsNone(non_existent)
    
    def test_verify_model(self):
        """Test verifying a model's integrity"""
        # Register a model
        model_metadata = {
            'model_hash': 'test_hash',
            'model_type': 'TestModel'
        }
        model_id = self.model_registry.register_model(model_metadata)
        
        # Verify with correct hash
        verified = self.model_registry.verify_model(model_id, 'test_hash')
        self.assertTrue(verified)
        
        # Verify with incorrect hash
        not_verified = self.model_registry.verify_model(model_id, 'wrong_hash')
        self.assertFalse(not_verified)
        
        # Verify with non-existent model ID
        non_existent = self.model_registry.verify_model('non_existent_id', 'test_hash')
        self.assertFalse(non_existent)
    
    def test_list_models(self):
        """Test listing all models in the registry"""
        # Register multiple models
        model_ids = []
        for i in range(3):
            model_metadata = {
                'model_hash': f'hash_{i}',
                'model_type': f'Model_{i}'
            }
            model_id = self.model_registry.register_model(model_metadata)
            model_ids.append(model_id)
        
        # List all models
        models = self.model_registry.list_models()
        
        # Check that all models are in the list
        self.assertEqual(len(models), 3)
        for model in models:
            self.assertIn(model['model_id'], model_ids)
    
    def test_record_model_update(self):
        """Test recording an update to a model"""
        # Register a model
        model_metadata = {
            'model_hash': 'original_hash',
            'model_type': 'TestModel'
        }
        model_id = self.model_registry.register_model(model_metadata)
        
        # Record an update
        update_info = {
            'update_type': 'parameter_change',
            'changes': {'param1': 'new_value'}
        }
        result = self.model_registry.record_model_update(model_id, update_info)
        
        # Check that the update was recorded
        self.assertTrue(result)
        model = self.model_registry.get_model(model_id)
        self.assertIn('history', model)
        self.assertEqual(len(model['history']), 1)
        self.assertEqual(model['history'][0]['update_type'], 'parameter_change')
        
        # Try to record an update for a non-existent model
        non_existent = self.model_registry.record_model_update('non_existent_id', update_info)
        self.assertFalse(non_existent)
    
    def test_get_model_history(self):
        """Test retrieving a model's history"""
        # Register a model
        model_metadata = {
            'model_hash': 'test_hash',
            'model_type': 'TestModel'
        }
        model_id = self.model_registry.register_model(model_metadata)
        
        # Record updates
        for i in range(3):
            update_info = {
                'update_type': f'update_{i}',
                'changes': {f'param_{i}': f'value_{i}'}
            }
            self.model_registry.record_model_update(model_id, update_info)
        
        # Get model history
        history = self.model_registry.get_model_history(model_id)
        
        # Check that history is correct
        self.assertEqual(len(history), 3)
        for i, update in enumerate(history):
            self.assertEqual(update['update_type'], f'update_{i}')
        
        # Get history for a model with no history
        new_model_id = self.model_registry.register_model({'model_hash': 'new_hash'})
        empty_history = self.model_registry.get_model_history(new_model_id)
        self.assertEqual(empty_history, [])
        
        # Get history for a non-existent model
        non_existent = self.model_registry.get_model_history('non_existent_id')
        self.assertEqual(non_existent, [])
    
    def test_update_model(self):
        """Test updating a model's metadata"""
        # Register a model
        model_metadata = {
            'model_hash': 'test_hash',
            'model_type': 'TestModel',
            'hyperparameters': {'param1': 'value1', 'param2': 'value2'}
        }
        model_id = self.model_registry.register_model(model_metadata)
        
        # Update metadata
        updated_metadata = {
            'model_hash': 'test_hash',  # Same hash
            'model_type': 'TestModel',
            'hyperparameters': {'param1': 'updated_value', 'param2': 'value2'},
            'description': 'Updated description'
        }
        
        result = self.model_registry.update_model(model_id, updated_metadata)
        
        # Check that the update was successful
        self.assertTrue(result)
        updated_model = self.model_registry.get_model(model_id)
        self.assertEqual(updated_model['hyperparameters']['param1'], 'updated_value')
        self.assertEqual(updated_model['description'], 'Updated description')
        
        # Try to update with a different hash (should raise an error)
        tampered_metadata = {
            'model_hash': 'different_hash',
            'model_type': 'TestModel'
        }
        
        with self.assertRaises(ValueError):
            self.model_registry.update_model(model_id, tampered_metadata)
        
        # Try to update a non-existent model
        non_existent = self.model_registry.update_model('non_existent_id', updated_metadata)
        self.assertFalse(non_existent)

if __name__ == '__main__':
    unittest.main()
