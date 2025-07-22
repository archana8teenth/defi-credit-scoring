#!/usr/bin/env python3
"""
Main script for DeFi Credit Scoring System
One-step execution script that generates wallet scores from JSON file
"""

import os
import sys
import argparse
import json
import pandas as pd
import logging
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor import AaveDataProcessor
from feature_engineer import DeFiFeatureEngineer  
from model_trainer import CreditScoringModel
from credit_scorer import DeFiCreditScorer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('credit_scoring.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories"""
    directories = ['models', 'data', 'results']
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)

def download_sample_data():
    """Download sample data if not available"""
    sample_data = [
        {
            "user": "0x1234567890123456789012345678901234567890",
            "action": "deposit",
            "amount": "1000.0",
            "timestamp": "2023-01-15T10:30:00Z",
            "token": "USDC"
        },
        {
            "user": "0x1234567890123456789012345678901234567890", 
            "action": "borrow",
            "amount": "500.0",
            "timestamp": "2023-01-16T14:20:00Z",
            "token": "ETH"
        },
        {
            "user": "0x1234567890123456789012345678901234567890",
            "action": "repay", 
            "amount": "250.0",
            "timestamp": "2023-01-20T09:15:00Z",
            "token": "ETH"
        },
        {
            "user": "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
            "action": "deposit",
            "amount": "2000.0", 
            "timestamp": "2023-01-10T16:45:00Z",
            "token": "DAI"
        },
        {
            "user": "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
            "action": "liquidationcall",
            "amount": "1500.0",
            "timestamp": "2023-01-25T11:30:00Z", 
            "token": "DAI"
        }
    ]
    
    # Generate more sample data
    import random
    from datetime import datetime, timedelta
    
    users = [
        "0x1234567890123456789012345678901234567890",
        "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd", 
        "0x1111111111111111111111111111111111111111",
        "0x2222222222222222222222222222222222222222",
        "0x3333333333333333333333333333333333333333"
    ]
    
    actions = ["deposit", "borrow", "repay", "withdraw", "liquidationcall"]
    tokens = ["USDC", "DAI", "ETH", "WBTC"]
    
    base_date = datetime(2023, 1, 1)
    
    for _ in range(100):  # Generate 100 more transactions
        user = random.choice(users)
        action = random.choice(actions)
        amount = round(random.uniform(10, 5000), 2)
        days_offset = random.randint(0, 365)
        timestamp = base_date + timedelta(days=days_offset)
        token = random.choice(tokens)
        
        sample_data.append({
            "user": user,
            "action": action, 
            "amount": str(amount),
            "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "token": token
        })
    
    with open('data/sample_transactions.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    logger.info("Generated sample transaction data")
    return 'data/sample_transactions.json'

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='DeFi Credit Scoring System')
    parser.add_argument('--input', '-i', type=str, help='Input JSON file path')
    parser.add_argument('--output', '-o', type=str, default='results/credit_scores.csv', help='Output CSV file path')
    parser.add_argument('--retrain', action='store_true', help='Force model retraining')
    
    args = parser.parse_args()
    
    # Create directories
    create_directories()
    
    # Input file
    if args.input and os.path.exists(args.input):
        input_file = args.input
    else:
        logger.info("No input file provided or file doesn't exist. Using sample data.")
        input_file = download_sample_data()
    
    try:
        # Step 1: Process Data
        logger.info("=== Step 1: Processing Transaction Data ===")
        processor = AaveDataProcessor()
        df = processor.process_data(input_file)
        
        print(f"\nData Summary:")
        print(f"- Total transactions: {len(df)}")
        print(f"- Unique wallets: {df['user'].nunique()}")
        print(f"- Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"- Action types: {df['action_category'].value_counts().to_dict()}")
        
        # Step 2: Feature Engineering
        logger.info("=== Step 2: Engineering Features ===")
        feature_engineer = DeFiFeatureEngineer()
        features = feature_engineer.engineer_features(df)
        
        print(f"\nFeature Summary:")
        print(f"- Features generated: {len(features.columns)}")
        print(f"- Wallets analyzed: {len(features)}")
        
        # Step 3: Train Models
        logger.info("=== Step 3: Training Models ===")
        model_trainer = CreditScoringModel()
        
        model_files_exist = os.path.exists('models/random_forest_model.pkl')
        
        if not model_files_exist or args.retrain:
            # Prepare training data
            X, y = model_trainer.prepare_training_data(features)
            
            # Train models
            training_results = model_trainer.train_models(X, y)
            
            # Save models
            model_trainer.save_models('models')
            
            print(f"\nTraining Results:")
            for name, results in training_results.items():
                print(f"- {name}: CV Score = {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
        else:
            logger.info("Loading existing models")
            model_trainer.load_models('models')
        
        # Step 4: Generate Credit Scores
        logger.info("=== Step 4: Generating Credit Scores ===")
        scorer = DeFiCreditScorer(model_trainer)
        scores = scorer.predict_scores(features)
        
        # Save results
        scores.to_csv(args.output)
        logger.info(f"Credit scores saved to {args.output}")
        
        # Step 5: Analysis and Summary
        logger.info("=== Step 5: Score Analysis ===")
        distribution = scorer.get_score_distribution()
        
        print(f"\nCredit Score Analysis:")
        print(f"- Total wallets scored: {distribution['total_wallets']}")
        print(f"- Mean score: {distribution['mean_score']:.1f}")
        print(f"- Median score: {distribution['median_score']:.1f}")
        print(f"- Standard deviation: {distribution['std_score']:.1f}")
        
        print(f"\nScore Distribution:")
        for range_name, count in distribution['score_ranges'].items():
            percentage = (count / distribution['total_wallets']) * 100
            print(f"- {range_name}: {count} wallets ({percentage:.1f}%)")
        
        print(f"\nRisk Categories:")
        for category, count in distribution['risk_categories'].items():
            percentage = (count / distribution['total_wallets']) * 100
            print(f"- {category}: {count} wallets ({percentage:.1f}%)")
        
        # Save analysis
        with open('results/score_analysis.json', 'w') as f:
            json.dump(distribution, f, indent=2)
        
        # Top and bottom wallets
        print(f"\nTop 5 Highest Scoring Wallets:")
        top_wallets = scores.nlargest(5, 'credit_score')[['credit_score', 'risk_category']]
        for wallet, row in top_wallets.iterrows():
            print(f"- {wallet}: {row['credit_score']} ({row['risk_category']})")
        
        print(f"\nTop 5 Lowest Scoring Wallets:")
        bottom_wallets = scores.nsmallest(5, 'credit_score')[['credit_score', 'risk_category']]
        for wallet, row in bottom_wallets.iterrows():
            print(f"- {wallet}: {row['credit_score']} ({row['risk_category']})")
        
        logger.info("Credit scoring complete!")
        
        return scores, distribution
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
