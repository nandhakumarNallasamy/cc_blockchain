#!/usr/bin/env python3
# Model Registry for tracking AI models in the blockchain system

import json
import os
import hashlib
from datetime import datetime
import uuid

class ModelRegistry:
    def __init__(self, registry_file="model_registry.json"):
        """Initialize the model registry"""
        self.registry_file = registry_file
        self.registry = {}
        self.load_registry()
    
    def load_registry(self):
        """Load the registry from file if it exists"""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {}
    
    def save_registry(self):
        """Save the registry to a file"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)
    
    def register_model(self, model_metadata):
        """Register a new model in the registry"""
        # Generate a unique model ID
        model_id = str(uuid.uuid4())
        
        # Add registration timestamp
        model_metadata['registered_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add the model to the registry
        self.registry[model_id] = model_metadata
        
        # Save changes
        self.save_registry()
        
        return model_id
    
    def get_model(self, model_id):
        """Get a model's metadata by ID"""
        return self.registry.get(model_id)
    
    def update_model(self, model_id, updated_metadata):
        """Update a model's metadata"""
        if model_id not in self.registry:
            return False
        
        # Get current metadata
        current_metadata = self.registry[model_id]
        
        # Verify model hash matches
        if current_metadata['model_hash'] != updated_metadata['model_hash']:
            raise ValueError("Model hash mismatch. Possible tampering detected.")
        
        # Update metadata
        current_metadata.update(updated_metadata)
        current_metadata['updated_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save changes
        self.save_registry()
        
        return True
    
    def verify_model(self, model_id, model_hash):
        """Verify a model's integrity by comparing hashes"""
        if model_id not in self.registry:
            return False
        
        registered_hash = self.registry[model_id]['model_hash']
        return registered_hash == model_hash
    
    def list_models(self):
        """List all registered models"""
        models = []
        for model_id, metadata in self.registry.items():
            model_info = metadata.copy()
            model_info['model_id'] = model_id
            models.append(model_info)
        
        return models
    
    def get_model_history(self, model_id):
        """Get the history of a model if tracked"""
        if model_id not in self.registry:
            return []
        
        model = self.registry[model_id]
        
        if 'history' not in model:
            return []
        
        return model['history']
    
    def record_model_update(self, model_id, update_info):
        """Record an update to a model in its history"""
        if model_id not in self.registry:
            return False
        
        model = self.registry[model_id]
        
        if 'history' not in model:
            model['history'] = []
        
        # Add timestamp to update info
        update_info['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add to history
        model['history'].append(update_info)
        
        # Save changes
        self.save_registry()
        
        return True
