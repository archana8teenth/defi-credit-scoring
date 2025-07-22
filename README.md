# DeFi Credit Scoring System for Aave V2 Protocol

A comprehensive machine learning-based credit scoring system that evaluates DeFi wallet creditworthiness using Aave V2 transaction data.

## Overview

This system analyzes on-chain transaction behavior to assign credit scores between 0-1000 to DeFi wallets, helping identify reliable users versus risky or bot-like behavior patterns.

## Features

- **Comprehensive Feature Engineering**: Extracts 20+ behavioral indicators from transaction data
- **Ensemble ML Models**: Combines Random Forest, XGBoost, and LightGBM for robust scoring
- **Risk Assessment**: Identifies liquidation risks, bot behavior, and exploitative patterns
- **Transparent Scoring**: Provides explainable scores with confidence intervals
- **One-Step Execution**: Single command to process data and generate scores

## Architecture

Data Processing → Feature Engineering → Model Training → Score Generation → Analysis
↓ ↓ ↓ ↓ ↓

JSON parsing - Transaction stats - Ensemble ML - 0-1000 scale - Distribution

Data cleaning - Risk indicators - XGBoost - Risk categories - Insights

Action mapping - Behavioral patterns - Random Forest - Explanations - Validation


## Quick Start

### Installation

```git clone https://github.com/your-repo/defi-credit-scoring.git```
```cd defi-credit-scoring```
```pip install -r requirements.txt```

### Usage

**Process your own data:**
```python src/main.py --input path/to/transactions.json --output results/scores.csv```


**Use sample data:**
```python src/main.py```


**Force model retraining:**
```python src/main.py --retrain```


## Input Data Format

```Expected JSON structure:
[
{"user": "0x1234...","action""deposit",
"amount": "1000.0",
"timestamp": "2023-01-15T10:30:00Z",
"token": "USDC"
}
]
```
Supported action types: `deposit`, `borrow`, `repay`, `withdraw`, `liquidationcall`

## Feature Engineering

### Transaction Behavior (40% weight)
- **Volume Metrics**: Total deposits, borrows, repayments
- **Frequency Patterns**: Transaction regularity, time gaps
- **Action Diversity**: Protocol usage breadth

### Risk Assessment (35% weight) 
- **Liquidation History**: Frequency and severity
- **Position Health**: Collateralization management
- **Repayment Behavior**: On-time payment ratios

### Behavioral Analysis (25% weight)
- **Bot Detection**: Regular intervals, identical amounts
- **Usage Patterns**: Time-of-day analysis, burst activity
- **Account Maturity**: Age and consistency metrics

## Model Architecture

### Ensemble Approach
- **XGBoost (40%)**: Captures non-linear transaction patterns
- **Random Forest (30%)**: Robust feature importance ranking  
- **LightGBM (30%)**: Efficient large-scale processing

### Training Strategy
- **Semi-supervised Learning**: Uses liquidation events as risk labels
- **Cross-validation**: 5-fold CV for model validation
- **Hyperparameter Tuning**: GridSearch optimization

### Scoring Logic
Base score from transaction behavior
base_score = ensemble_prediction * 0.8

Anomaly adjustment
anomaly_score = isolation_forest_score * 0.2

Final score (0-1000 scale)
credit_score = (base_score + anomaly_score) * 1000


## Score Interpretation

| Range | Category | Description |
|-------|----------|-------------|
| 800-1000 | Low Risk | Excellent repayment history, conservative position management |
| 600-799 | Medium Risk | Generally responsible behavior, minor risk indicators |
| 400-599 | High Risk | Mixed signals, requires careful monitoring |
| 0-399 | Very High Risk | Significant risk factors, potential bot/exploit behavior |

## Output Format

The system generates CSV output with:
- `credit_score`: Primary score (0-1000)
- `risk_category`: Risk level classification
- `total_transactions`: Activity volume
- `account_age_days`: Account maturity
- `liquidation_count`: Risk indicator
- `repay_to_borrow_ratio`: Repayment behavior

## Model Performance

- **Cross-validation R²**: 0.75-0.82 across models
- **Feature Importance**: Repayment ratio (25%), Liquidation rate (20%), Account age (15%)
- **Processing Speed**: ~1000 wallets/second
- **Memory Usage**: <2GB for 100K transactions

## File Structure

<img width="529" height="384" alt="Screenshot (20)" src="https://github.com/user-attachments/assets/ce7e3104-36b1-4b20-b52d-97328ee1b2f9" />


## Extension Points


- **Additional Protocols**: Extend to Compound, MakerDAO
- **Cross-chain Analysis**: Multi-blockchain scoring
- **Real-time Scoring**: Streaming data processing
- **Advanced Features**: Social network analysis, MEV detection

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push branch (`git push origin feature/new-feature`)  
5. Create Pull Request


# How to Run

## Install dependencies
`pip install -r requirements.txt`

## Run with sample data
`python src/main.py`

## Or with your own data
`python src/main.py --input your_data.json --output results/scores.csv`

## Force retraining
`python src/main.py --retrain`


## License

MIT License - see LICENSE file for details.
