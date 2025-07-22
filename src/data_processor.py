import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AaveDataProcessor:
    """
    Process Aave V2 transaction data from JSON format
    """
    
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        
    def load_json_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and parse JSON transaction data
        
        Args:
            file_path (str): Path to JSON file
            
        Returns:
            pd.DataFrame: Processed transaction data
        """
        logger.info(f"Loading data from {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Handle nested JSON structure
                df = pd.json_normalize(data)
            else:
                raise ValueError("Unsupported JSON format")
                
            logger.info(f"Loaded {len(df)} transactions")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate transaction data
        
        Args:
            df (pd.DataFrame): Raw transaction data
            
        Returns:
            pd.DataFrame: Cleaned transaction data
        """
        logger.info("Cleaning and validating data")
        
        # Essential columns we expect in Aave data
        required_columns = ['user', 'action', 'amount', 'timestamp']
        
        # Check for required columns (flexible naming)
        available_columns = df.columns.tolist()
        column_mapping = {}
        
        for req_col in required_columns:
            # Find matching column (case insensitive, flexible naming)
            matches = [col for col in available_columns if req_col.lower() in col.lower()]
            if matches:
                column_mapping[req_col] = matches[0]
            elif req_col == 'user':
                # Look for wallet/address columns
                wallet_matches = [col for col in available_columns if 
                                any(term in col.lower() for term in ['user', 'wallet', 'address', 'account'])]
                if wallet_matches:
                    column_mapping[req_col] = wallet_matches[0]
            elif req_col == 'action':
                # Look for action/type columns
                action_matches = [col for col in available_columns if 
                                any(term in col.lower() for term in ['action', 'type', 'method', 'function'])]
                if action_matches:
                    column_mapping[req_col] = action_matches[0]
        
        # Rename columns to standard names
        df = df.rename(columns=column_mapping)
        
        # Handle missing required columns
        for req_col in required_columns:
            if req_col not in df.columns:
                if req_col == 'timestamp':
                    df['timestamp'] = pd.Timestamp.now()
                elif req_col == 'amount':
                    df['amount'] = 1.0  # Default amount
                else:
                    logger.warning(f"Missing required column: {req_col}")
                    df[req_col] = 'unknown'
        
        # Data cleaning
        df = df.dropna(subset=['user'])  # Remove rows without user
        
        # Clean user addresses (convert to lowercase for consistency)
        df['user'] = df['user'].astype(str).str.lower().str.strip()
        
        # Clean action types
        df['action'] = df['action'].astype(str).str.lower().str.strip()
        
        # Convert amounts to numeric
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            df['amount'] = df['amount'].fillna(0)
        
        # Convert timestamps
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df['timestamp'] = df['timestamp'].fillna(pd.Timestamp.now())
            except:
                df['timestamp'] = pd.Timestamp.now()
        
        # Add derived columns
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        logger.info(f"Cleaned data: {len(df)} transactions, {df['user'].nunique()} unique users")
        
        return df
    
    def categorize_actions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize transaction actions into standard types
        
        Args:
            df (pd.DataFrame): Transaction data
            
        Returns:
            pd.DataFrame: Data with categorized actions
        """
        # Standard Aave action mapping
        action_mapping = {
            'deposit': ['deposit', 'supply', 'lend'],
            'borrow': ['borrow'],
            'repay': ['repay', 'repayment'],
            'withdraw': ['withdraw', 'redeem', 'redeemunderlying'],
            'liquidation': ['liquidationcall', 'liquidate', 'liquidation'],
            'flashloan': ['flashloan', 'flash'],
        }
        
        def map_action(action):
            action = str(action).lower()
            for standard_action, variants in action_mapping.items():
                if any(variant in action for variant in variants):
                    return standard_action
            return 'other'
        
        df['action_category'] = df['action'].apply(map_action)
        
        return df
    
    def process_data(self, file_path: str) -> pd.DataFrame:
        """
        Complete data processing pipeline
        
        Args:
            file_path (str): Path to JSON file
            
        Returns:
            pd.DataFrame: Processed transaction data
        """
        # Load data
        df = self.load_json_data(file_path)
        
        # Clean and validate
        df = self.clean_and_validate_data(df)
        
        # Categorize actions
        df = self.categorize_actions(df)
        
        self.processed_data = df
        
        logger.info("Data processing complete")
        return df
