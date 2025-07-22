import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DeFiCreditScorer:
    """
    Generate credit scores from trained models
    """
    
    def __init__(self, model_trainer):
        self.model_trainer = model_trainer
        self.scores = None
        
    def predict_scores(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict credit scores using trained models
        
        Args:
            features (pd.DataFrame): Feature matrix
            
        Returns:
            pd.DataFrame: Credit scores and explanations
        """
        logger.info("Generating credit scores")
        
        if not self.model_trainer.is_trained:
            raise ValueError("Models must be trained before scoring")
        
        # Prepare features
        selected_features = self.model_trainer.select_features(features)
        X = features[selected_features].fillna(0)
        
        # Scale features
        scaler = self.model_trainer.scalers['main']
        X_scaled = pd.DataFrame(
            scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Predict with each model
        predictions = {}
        for name, model in self.model_trainer.models.items():
            if name != 'isolation_forest':  # Skip anomaly detection model
                try:
                    pred = model.predict(X_scaled)
                    predictions[name] = np.clip(pred, 0, 1)
                except Exception as e:
                    logger.warning(f"Error predicting with {name}: {str(e)}")
                    continue
        
        # Anomaly scores
        if 'isolation_forest' in self.model_trainer.models:
            anomaly_scores = self.model_trainer.models['isolation_forest'].decision_function(X_scaled)
            # Normalize to 0-1 (higher is better)
            anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
        else:
            anomaly_scores = np.ones(len(X))
        
        # Ensemble prediction (weighted average)
        if predictions:
            weights = {
                'random_forest': 0.3,
                'xgboost': 0.4,
                'lightgbm': 0.3
            }
            
            ensemble_score = np.zeros(len(X))
            total_weight = 0
            
            for name, pred in predictions.items():
                weight = weights.get(name, 0.33)
                ensemble_score += pred * weight
                total_weight += weight
            
            if total_weight > 0:
                ensemble_score /= total_weight
        else:
            logger.warning("No valid predictions, using default scores")
            ensemble_score = np.full(len(X), 0.5)
        
        # Combine with anomaly scores
        final_score = (ensemble_score * 0.8) + (anomaly_scores * 0.2)
        
        # Convert to 0-1000 scale
        credit_scores = np.clip(final_score * 1000, 0, 1000)
        
        # Create results dataframe
        results = pd.DataFrame(index=features.index)
        results['credit_score'] = credit_scores.round(0).astype(int)
        results['risk_category'] = self._categorize_risk(credit_scores)
        
        # Add individual model scores for transparency
        for name, pred in predictions.items():
            results[f'{name}_score'] = (pred * 1000).round(0).astype(int)
        
        results['anomaly_score'] = (anomaly_scores * 1000).round(0).astype(int)
        
        # Add explanatory features
        results['total_transactions'] = features.get('total_transactions', 0)
        results['account_age_days'] = features.get('account_age_days', 0)
        results['liquidation_count'] = features.get('liquidation_count', 0)
        results['repay_to_borrow_ratio'] = features.get('repay_to_borrow_ratio', 0)
        
        self.scores = results
        logger.info(f"Generated credit scores for {len(results)} wallets")
        
        return results
    
    def _categorize_risk(self, scores: np.ndarray) -> List[str]:
        """
        Categorize risk levels based on scores
        
        Args:
            scores (np.ndarray): Credit scores (0-1000)
            
        Returns:
            List[str]: Risk categories
        """
        categories = []
        for score in scores:
            if score >= 800:
                categories.append('Low Risk')
            elif score >= 600:
                categories.append('Medium Risk')
            elif score >= 400:
                categories.append('High Risk')
            else:
                categories.append('Very High Risk')
        
        return categories
    
    def explain_score(self, wallet_address: str) -> Dict:
        """
        Provide explanation for a specific wallet's score
        
        Args:
            wallet_address (str): Wallet address
            
        Returns:
            Dict: Score explanation
        """
        if self.scores is None:
            raise ValueError("Scores must be generated before explanation")
        
        if wallet_address not in self.scores.index:
            raise ValueError(f"Wallet {wallet_address} not found in scored wallets")
        
        wallet_data = self.scores.loc[wallet_address]
        
        explanation = {
            'wallet': wallet_address,
            'credit_score': int(wallet_data['credit_score']),
            'risk_category': wallet_data['risk_category'],
            'key_factors': {
                'total_transactions': int(wallet_data['total_transactions']),
                'account_age_days': int(wallet_data['account_age_days']),
                'liquidation_count': int(wallet_data['liquidation_count']),
                'repay_to_borrow_ratio': float(wallet_data['repay_to_borrow_ratio'])
            }
        }
        
        # Score interpretation
        if wallet_data['credit_score'] >= 800:
            explanation['interpretation'] = "Excellent credit profile with strong repayment history and responsible DeFi usage"
        elif wallet_data['credit_score'] >= 600:
            explanation['interpretation'] = "Good credit profile with generally responsible behavior"
        elif wallet_data['credit_score'] >= 400:
            explanation['interpretation'] = "Fair credit profile with some risk indicators"
        else:
            explanation['interpretation'] = "Poor credit profile with significant risk factors"
        
        return explanation
    
    def get_score_distribution(self) -> Dict:
        """
        Get distribution of credit scores
        
        Returns:
            Dict: Score distribution statistics
        """
        if self.scores is None:
            raise ValueError("Scores must be generated first")
        
        scores = self.scores['credit_score']
        
        distribution = {
            'total_wallets': len(scores),
            'mean_score': float(scores.mean()),
            'median_score': float(scores.median()),
            'std_score': float(scores.std()),
            'score_ranges': {
                '0-100': len(scores[(scores >= 0) & (scores < 100)]),
                '100-200': len(scores[(scores >= 100) & (scores < 200)]),
                '200-300': len(scores[(scores >= 200) & (scores < 300)]),
                '300-400': len(scores[(scores >= 300) & (scores < 400)]),
                '400-500': len(scores[(scores >= 400) & (scores < 500)]),
                '500-600': len(scores[(scores >= 500) & (scores < 600)]),
                '600-700': len(scores[(scores >= 600) & (scores < 700)]),
                '700-800': len(scores[(scores >= 700) & (scores < 800)]),
                '800-900': len(scores[(scores >= 800) & (scores < 900)]),
                '900-1000': len(scores[(scores >= 900) & (scores <= 1000)])
            },
            'risk_categories': self.scores['risk_category'].value_counts().to_dict()
        }
        
        return distribution
