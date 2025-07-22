import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class DeFiFeatureEngineer:
    """
    Extract features from DeFi transaction data for credit scoring
    """
    
    def __init__(self):
        self.features = None
        
    def calculate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic transaction statistics per wallet
        
        Args:
            df (pd.DataFrame): Transaction data
            
        Returns:
            pd.DataFrame: Basic features per wallet
        """
        logger.info("Calculating basic features")
        
        # Group by user
        user_stats = df.groupby('user').agg({
            'timestamp': ['count', 'min', 'max'],
            'amount': ['sum', 'mean', 'std', 'min', 'max'],
            'action_category': lambda x: x.nunique()
        }).round(6)
        
        # Flatten column names
        user_stats.columns = [f"{col[0]}_{col[1]}" for col in user_stats.columns]
        
        # Rename columns
        user_stats = user_stats.rename(columns={
            'timestamp_count': 'total_transactions',
            'timestamp_min': 'first_transaction',
            'timestamp_max': 'last_transaction',
            'amount_sum': 'total_volume',
            'amount_mean': 'avg_transaction_size',
            'amount_std': 'transaction_size_std',
            'amount_min': 'min_transaction_size',
            'amount_max': 'max_transaction_size',
            'action_category_<lambda>': 'unique_actions'
        })
        
        # Account age in days
        user_stats['account_age_days'] = (
            user_stats['last_transaction'] - user_stats['first_transaction']
        ).dt.days
        user_stats['account_age_days'] = user_stats['account_age_days'].fillna(0)
        
        # Transaction frequency
        user_stats['avg_transactions_per_day'] = (
            user_stats['total_transactions'] / (user_stats['account_age_days'] + 1)
        )
        
        return user_stats
    
    def calculate_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features based on action types
        
        Args:
            df (pd.DataFrame): Transaction data
            
        Returns:
            pd.DataFrame: Action-based features per wallet
        """
        logger.info("Calculating action-based features")
        
        # Action counts and volumes
        action_features = df.groupby(['user', 'action_category']).agg({
            'timestamp': 'count',
            'amount': 'sum'
        }).unstack(fill_value=0)
        
        # Flatten columns
        action_features.columns = [f"{col[1]}_{col[0]}" for col in action_features.columns]
        
        # Calculate ratios
        total_transactions = action_features.filter(regex='_count$').sum(axis=1)
        total_volume = action_features.filter(regex='_sum$').sum(axis=1)
        
        # Repayment behavior
        if 'repay_count' in action_features.columns and 'borrow_count' in action_features.columns:
            action_features['repay_to_borrow_ratio'] = (
                action_features['repay_count'] / (action_features['borrow_count'] + 1)
            )
            action_features['repay_volume_ratio'] = (
                action_features['repay_sum'] / (action_features['borrow_sum'] + 1)
            )
        else:
            action_features['repay_to_borrow_ratio'] = 0
            action_features['repay_volume_ratio'] = 0
        
        # Liquidation risk
        if 'liquidation_count' in action_features.columns:
            action_features['liquidation_rate'] = (
                action_features['liquidation_count'] / total_transactions
            )
        else:
            action_features['liquidation_count'] = 0
            action_features['liquidation_rate'] = 0
        
        # Activity diversity
        action_features['action_diversity'] = (action_features > 0).sum(axis=1)
        
        return action_features
    
    def calculate_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk-related features
        
        Args:
            df (pd.DataFrame): Transaction data
            
        Returns:
            pd.DataFrame: Risk features per wallet
        """
        logger.info("Calculating risk features")
        
        risk_features = []
        
        for user, user_data in df.groupby('user'):
            user_data = user_data.sort_values('timestamp')
            
            features = {
                'user': user,
                'has_liquidations': 1 if 'liquidation' in user_data['action_category'].values else 0,
                'liquidation_frequency': (user_data['action_category'] == 'liquidation').sum(),
                'time_between_transactions_std': 0,
                'regular_activity_score': 0,
                'late_night_activity_ratio': 0,
                'weekend_activity_ratio': 0,
            }
            
            if len(user_data) > 1:
                # Time between transactions
                time_diffs = user_data['timestamp'].diff().dt.total_seconds() / 3600  # hours
                time_diffs = time_diffs.dropna()
                if len(time_diffs) > 0:
                    features['time_between_transactions_std'] = time_diffs.std()
                    features['regular_activity_score'] = 1 / (1 + time_diffs.std() / time_diffs.mean()) if time_diffs.mean() > 0 else 0
            
            # Activity timing analysis
            if 'hour' in user_data.columns:
                late_night_txns = ((user_data['hour'] >= 0) & (user_data['hour'] <= 6)).sum()
                features['late_night_activity_ratio'] = late_night_txns / len(user_data)
            
            if 'day_of_week' in user_data.columns:
                weekend_txns = (user_data['day_of_week'].isin([5, 6])).sum()
                features['weekend_activity_ratio'] = weekend_txns / len(user_data)
            
            risk_features.append(features)
        
        return pd.DataFrame(risk_features).set_index('user')
    
    def calculate_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate behavioral pattern features
        
        Args:
            df (pd.DataFrame): Transaction data
            
        Returns:
            pd.DataFrame: Behavioral features per wallet
        """
        logger.info("Calculating behavioral features")
        
        behavioral_features = []
        
        for user, user_data in df.groupby('user'):
            features = {
                'user': user,
                'bot_like_score': 0,
                'amount_variance': 0,
                'transaction_regularity': 0,
                'burst_activity_score': 0,
            }
            
            if len(user_data) > 1:
                amounts = user_data['amount'].values
                
                # Bot-like behavior detection
                # Check for repeated exact amounts
                unique_amounts = len(np.unique(amounts))
                features['amount_variance'] = unique_amounts / len(amounts)
                
                # Regular intervals (potential bot behavior)
                if len(user_data) > 2:
                    time_diffs = user_data.sort_values('timestamp')['timestamp'].diff().dt.total_seconds()
                    time_diffs = time_diffs.dropna()
                    if len(time_diffs) > 0:
                        # Check for suspiciously regular intervals
                        cv = time_diffs.std() / time_diffs.mean() if time_diffs.mean() > 0 else 1
                        features['transaction_regularity'] = 1 / (1 + cv)  # Lower CV = more regular
                
                # Burst activity detection
                user_data_sorted = user_data.sort_values('timestamp')
                daily_counts = user_data_sorted.groupby(user_data_sorted['timestamp'].dt.date).size()
                if len(daily_counts) > 0:
                    features['burst_activity_score'] = daily_counts.max() / daily_counts.mean()
            
            behavioral_features.append(features)
        
        return pd.DataFrame(behavioral_features).set_index('user')
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline
        
        Args:
            df (pd.DataFrame): Processed transaction data
            
        Returns:
            pd.DataFrame: Feature matrix for all wallets
        """
        logger.info("Starting feature engineering")
        
        # Calculate different feature sets
        basic_features = self.calculate_basic_features(df)
        action_features = self.calculate_action_features(df)
        risk_features = self.calculate_risk_features(df)
        behavioral_features = self.calculate_behavioral_features(df)
        
        # Combine all features
        features = basic_features.join([action_features, risk_features, behavioral_features], how='outer')
        features = features.fillna(0)
        
        # Add derived features
        features['volume_per_transaction'] = features['total_volume'] / features['total_transactions']
        features['volume_per_transaction'] = features['volume_per_transaction'].fillna(0)
        
        # Risk score components
        features['position_health_score'] = (
            features['repay_to_borrow_ratio'] * 0.4 +
            (1 - features['liquidation_rate']) * 0.6
        )
        
        # Normalize some features
        for col in ['total_volume', 'avg_transaction_size', 'max_transaction_size']:
            if col in features.columns:
                features[f'{col}_log'] = np.log1p(features[col])
        
        self.features = features
        logger.info(f"Feature engineering complete. Generated {len(features.columns)} features for {len(features)} wallets")
        
        return features
