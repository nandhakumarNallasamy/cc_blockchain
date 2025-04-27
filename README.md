# Blockchain-based Credit Card Fraud Detection System

This project implements a blockchain-based framework for securing AI-powered credit card fraud detection systems, as described in the research reports.

## Project Overview

The system addresses key vulnerabilities in traditional fraud detection systems:

- **Model Tampering:** Prevents unauthorized modifications to fraud detection models
- **Audit Trails:** Creates immutable records of model updates and fraud determinations
- **Verification:** Enables cryptographic verification of model integrity
- **Decision Transparency:** Records model decisions for accountability

## Components

The system consists of the following components:

1. **Blockchain Implementation** - A basic blockchain implementation for recording transactions and maintaining an immutable ledger
2. **Model Registry** - A secure registry for tracking AI model versions and metadata
3. **Fraud Detection Models** - Machine learning models trained on credit card transaction data
4. **Secure Deployment System** - Framework for deploying models with integrity verification
5. **API Server** - RESTful API for interacting with the system

## Files

- `main.py` - Main application script that demonstrates the system's functionality
- `blockchain.py` - Implementation of the blockchain for recording transactions
- `model_registry.py` - System for registering and verifying model integrity
- `prepare_dataset.py` - Script to prepare the credit card fraud dataset
- `model_training.py` - Trains and evaluates fraud detection models
- `secure_model_deployment.py` - Manages secure model deployment and verification
- `api_server.py` - Simple REST API for interacting with the system
- `transaction_processor.py` - Real-time transaction processing with blockchain validation
- `demo.py` - Demonstration script showing the system in action
- `test_blockchain.py` - Tests for the blockchain component
- `test_model_registry.py` - Tests for the model registry

## Dataset

The system uses the Credit Card Fraud Detection dataset from Kaggle, which contains:
- 284,807 transactions from European cardholders
- 492 fraudulent transactions (0.172% of the total)
- 28 anonymized features (V1-V28) for privacy protection
- Original transaction amount and time variables
- Class labels indicating fraud (1) or legitimate (0) transactions

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- Required packages:
  - pandas
  - numpy
  - scikit-learn
  - imbalanced-learn
  - matplotlib
  - seaborn
  - flask
  - joblib

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Prepare the Dataset

1. Download the Credit Card Fraud Detection dataset from Kaggle and save it as `creditcard.csv`
2. Run the data preparation script:

```bash
python prepare_dataset.py
```

### Train Models

```bash
python model_training.py
```

## Usage

### Run the Main Demo

```bash
python main.py
```

This will:
1. Load and preprocess the dataset
2. Train a fraud detection model
3. Register the model on the blockchain
4. Simulate transaction processing
5. Verify model integrity
6. Simulate a model tampering attack and detect it

### Secure Model Deployment

```bash
python secure_model_deployment.py results/RandomForest_model.pkl creditcard.csv
```

### Start the API Server

```bash
python api_server.py
```

## API Endpoints

- `GET /health` - Check system health
- `POST /load_model` - Load and register a model
- `POST /predict` - Make a fraud prediction
- `GET /verify_model` - Verify model integrity
- `GET /blockchain/status` - Get blockchain status
- `GET /transaction/verify` - Verify a transaction

## Blockchain System

The blockchain implementation provides:

1. **Immutable Audit Trail** - All model registrations, updates, and fraud determinations are permanently recorded
2. **Tamper Detection** - Any unauthorized changes to the model are immediately detected
3. **Multi-Party Verification** - Different organizations can participate in the validation process
4. **Decision Transparency** - All fraud determinations are recorded with their decision parameters

## Security Features

- **Cryptographic Model Verification** - Models are hashed to ensure integrity
- **Consensus-Based Updates** - Changes require multi-party agreement
- **Immutable Decision Records** - All fraud determinations are permanently recorded
- **Transparent Audit System** - Complete history of model operations is available

## Architecture

### Blockchain Implementation

The blockchain is implemented as a chain of blocks, where each block contains:
- A list of transactions
- A timestamp
- A hash of the previous block
- A proof-of-work nonce

The blockchain uses a simple proof-of-work consensus mechanism that requires finding a hash with a specified number of leading zeros.

### Model Registry

The model registry maintains a secure record of all models, including:
- Model metadata (type, hyperparameters)
- Model hash for integrity verification
- Version history and update records
- Performance metrics

### Fraud Detection Model

The system uses a Random Forest classifier trained on the credit card fraud dataset. The model is evaluated using:
- Precision and recall metrics (critical for imbalanced fraud detection)
- ROC and Precision-Recall curves
- Confusion matrix analysis

### Integration Architecture

The components are integrated as follows:
1. Models are trained and registered in the model registry
2. Model metadata and hash are recorded on the blockchain
3. Transaction validation uses the registered model
4. Validation decisions are recorded on the blockchain
5. Regular integrity checks detect tampering attempts

## Organization Structure

The blockchain network includes the following organizations:
1. **Payment Processor** - Responsible for transaction validation
2. **Financial Institution** - Provides transaction data and verifies models
3. **Regulatory Authority** - Oversees the system and approves model updates
4. **Model Development Team** - Creates and updates fraud detection models

## Future Improvements

- Implement full permissioned blockchain network with multiple organizations
- Add support for federated learning across institutions
- Improve privacy-preserving techniques for sensitive transaction data
- Develop more sophisticated consensus mechanisms for model updates
- Create comprehensive visualization dashboard for system monitoring

## Running Tests

```bash
# Run blockchain tests
python -m unittest test_blockchain.py

# Run model registry tests
python -m unittest test_model_registry.py
```

## Real-time Transaction Processing

For real-time transaction processing, use the transaction processor:

```bash
python transaction_processor.py model_path [csv_file]
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT
