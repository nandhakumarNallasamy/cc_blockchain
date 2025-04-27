# Blockchain-based Credit Card Fraud Detection System

## Project Overview

The system addresses key vulnerabilities in traditional fraud detection systems:

- **Model Tampering:** Prevents unauthorized modifications to fraud detection models
- **Audit Trails:** Creates immutable records of model updates and fraud determinations
- **Verification:** Enables cryptographic verification of model integrity
- **Decision Transparency:** Records model decisions for accountability

## Dataset Download

**IMPORTANT:** The credit card fraud detection dataset is not included in this repository due to its large size.

### How to Get the Dataset
1. Visit the Kaggle dataset page: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Download the `creditcard.csv` file
3. Place the `creditcard.csv` file in the project's root directory

### Dataset Details
- 284,807 transactions from European cardholders
- 492 fraudulent transactions (0.172% of the total)
- 28 anonymized features (V1-V28) for privacy protection
- Original transaction amount and time variables
- Class labels indicating fraud (1) or legitimate (0) transactions

## Components

The system consists of the following components:

1. **Blockchain Implementation** - A basic blockchain implementation for recording transactions and maintaining an immutable ledger
2. **Model Registry** - A secure registry for tracking AI model versions and metadata
3. **Fraud Detection Models** - Machine learning models trained on credit card transaction data
4. **Secure Deployment System** - Framework for deploying models with integrity verification
5. **API Server** - RESTful API for interacting with the system

## Execution Guide

### Prerequisites
- Python 3.8 or higher
- Kaggle account (to download the dataset)

### Installation Steps
1. Clone the repository
```bash
git clone https://github.com/yourusername/blockchain-fraud-detection.git
cd blockchain-fraud-detection
```

2. Create a virtual environment (optional but recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

### Workflow

#### 1. Prepare Dataset
```bash
python prepare_dataset.py
```
This script will:
- Load the dataset
- Preprocess the data
- Split into training and testing sets
- Generate visualizations

#### 2. Train Models
```bash
python model_training.py
```
This will:
- Train multiple fraud detection models
- Evaluate model performance
- Save best-performing models
- Generate performance reports and visualizations

#### 3. Demonstration
```bash
python demo.py
```
A comprehensive demo showing:
- Model registration
- Blockchain-based transaction processing
- Fraud detection
- Model integrity verification
- Tampering detection

#### 4. Real-time Transaction Processing
```bash
python transaction_processor.py results/RandomForest_model.pkl creditcard.csv
```

### Running Tests
```bash
# Blockchain tests
python -m unittest test_blockchain.py

# Model registry tests
python -m unittest test_model_registry.py
```

## Blockchain System

The blockchain implementation provides:

1. **Immutable Audit Trail** - All model registrations, updates, and fraud determinations are permanently recorded
2. **Tamper Detection** - Any unauthorized changes to the model are immediately detected
3. **Multi-Party Verification** - Different organizations can participate in the validation process
4. **Decision Transparency** - All fraud determinations are recorded with their decision parameters

## Additional Execution Options

### API Server
```bash
python api_server.py
```
Starts a Flask API for model prediction and blockchain interaction.

### Secure Model Deployment
```bash
python secure_model_deployment.py results/RandomForest_model.pkl creditcard.csv
```

## Organizational Structure

The blockchain network includes the following organizations:
1. **Payment Processor** - Responsible for transaction validation
2. **Financial Institution** - Provides transaction data and verifies models
3. **Regulatory Authority** - Oversees the system and approves model updates
4. **Model Development Team** - Creates and updates fraud detection models

## Security Features

- **Cryptographic Model Verification** - Models are hashed to ensure integrity
- **Consensus-Based Updates** - Changes require multi-party agreement
- **Immutable Decision Records** - All fraud determinations are permanently recorded
- **Transparent Audit System** - Complete history of model operations is available

## Future Improvements

- Implement full permissioned blockchain network with multiple organizations
- Add support for federated learning across institutions
- Improve privacy-preserving techniques for sensitive transaction data
- Develop more sophisticated consensus mechanisms for model updates
- Create comprehensive visualization dashboard for system monitoring

## Troubleshooting
- Ensure you have downloaded `creditcard.csv` from Kaggle
- Check that all dependencies are installed
- Verify Python version compatibility

## Contributing
Contributions are welcome! Please submit pull requests or open issues.

## License
MIT License - See LICENSE file for details
