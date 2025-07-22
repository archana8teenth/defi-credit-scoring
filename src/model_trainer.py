import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
import joblib
import logging

logger = logging.getLogger(__name__)

class CreditScoringModel:
    """
    Machine Learning model for DeFi credit scoring
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.is_trained = False
        
    def prepare_training_data(self, features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data with pseudo-labels for unsupervised learning
        
        Args:
            features (pd.DataFrame): Feature matrix
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and pseudo-labels
        """
        logger.info("Preparing training data")
        
        # Create pseudo-labels based on risk indicators
        risk_score = 0
        
        # Positive indicators (increase creditworthiness)
        if 'position_health_score' in features.columns:
            risk_score += features['position_health_score'] * 0.3
        
        if 'repay_to_borrow_ratio' in features.columns:
            risk_score += np.clip(features['repay_to_borrow_ratio'], 0, 2) * 0.2
        
        if 'account_age_days' in features.columns:
            risk_score += np.clip(features['account_age_days'] / 365, 0, 2) * 0.1
        
        if 'action_diversity' in features.columns:
            risk_score += np.clip(features['action_diversity'] / 5, 0, 1) * 0.1
        
        # Negative indicators (decrease creditworthiness)
        if 'liquidation_rate' in features.columns:
            risk_score -= features['liquidation_rate'] * 0.4
        
        if 'bot_like_score' in features.columns:
            risk_score -= features['bot_like_score'] * 0.2
        
        if 'transaction_regularity' in features.columns:
            # Very high regularity might indicate bot behavior
            high_regularity_penalty = np.where(features['transaction_regularity'] > 0.9, 0.1, 0)
            risk_score -= high_regularity_penalty
        
        # Normalize to 0-1 range
        risk_score = np.clip(risk_score, 0, 1)
        
        # Add some noise for better model training
        noise = np.random.normal(0, 0.05, len(risk_score))
        pseudo_labels = np.clip(risk_score + noise, 0, 1)
        
        return features, pd.Series(pseudo_labels, index=features.index)
    
    def select_features(self, features: pd.DataFrame) -> List[str]:
        """
        Select relevant features for training
        
        Args:
            features (pd.DataFrame): Feature matrix
            
        Returns:
            List[str]: Selected feature names
        """
        # Remove non-numeric features and handle infinite values
        numeric_features = features.select_dtypes(include=[np.number]).columns
        
        # Remove features with too many missing values or infinite values
        valid_features = []
        for col in numeric_features:
            if not features[col].isin([np.inf, -np.inf]).any() and features[col].notna().sum() > len(features) * 0.5:
                valid_features.append(col)
        
        return valid_features
    
    def train_models(self, features: pd.DataFrame, labels: pd.Series) -> Dict:
        """
        Train ensemble of models
        
        Args:
            features (pd.DataFrame): Feature matrix
            labels (pd.Series): Target labels
            
        Returns:
            Dict: Training results
        """
        logger.info("Training models")
        
        # Select features
        selected_features = self.select_features(features)
        X = features[selected_features].fillna(0)
        y = labels
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        self.scalers['main'] = scaler
        
        # Train multiple models
        models_config = {
            'random_forest': {
                'model': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                'params': {'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10]}
            },
            'xgboost': {
                'model': xgb.XGBRegressor(random_state=42, n_jobs=-1),
                'params': {'max_depth': [6, 10], 'learning_rate': [0.1, 0.2], 'n_estimators': [100, 200]}
            },
            'lightgbm': {
                'model': lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
                'params': {'max_depth': [10, 20], 'learning_rate': [0.1, 0.2], 'n_estimators': [100, 200]}
            }
        }
        
        results = {}
        
        for name, config in models_config.items():
            logger.info(f"Training {name}")
            
            try:
                # Grid search for best parameters
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=3,
                    scoring='r2',
                    n_jobs=-1
                )
                
                grid_search.fit(X_scaled, y)
                
                # Store best model
                best_model = grid_search.best_estimator_
                self.models[name] = best_model
                
                # Calculate feature importance
                if hasattr(best_model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': X_scaled.columns,
                        'importance': best_model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    self.feature_importance[name] = importance_df
                
                # Cross-validation score
                cv_scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring='r2')
                
                results[name] = {
                    'model': best_model,
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                logger.info(f"{name} - Best score: {grid_search.best_score_:.4f}, CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                continue
        
        # Train anomaly detection model
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_scores = isolation_forest.fit_predict(X_scaled)
        self.models['isolation_forest'] = isolation_forest
        
        self.is_trained = True
        logger.info("Model training complete")
        
        return results
    
    def save_models(self, model_dir: str):
        """
        Save trained models
        
        Args:
            model_dir (str): Directory to save models
        """
        if not self.is_trained:
            logger.warning("No trained models to save")
            return
        
        logger.info(f"Saving models to {model_dir}")
        
        for name, model in self.models.items():
            joblib.dump(model, f"{model_dir}/{name}_model.pkl")
        
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f"{model_dir}/{name}_scaler.pkl")
        
        # Save feature importance
        for name, importance in self.feature_importance.items():
            importance.to_csv(f"{model_dir}/{name}_feature_importance.csv", index=False)
    
    def load_models(self, model_dir: str):
        """
        Load pre-trained models
        
        Args:
            model_dir (str): Directory containing saved models
        """
        import os
        
        logger.info(f"Loading models from {model_dir}")
        
        model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.pkl')]
        scaler_files = [f for f in os.listdir(model_dir) if f.endswith('_scaler.pkl')]
        
        for file in model_files:
            model_name = file.replace('_model.pkl', '')
            self.models[model_name] = joblib.load(f"{model_dir}/{file}")
        
        for file in scaler_files:
            scaler_name = file.replace('_scaler.pkl', '')
            self.scalers[scaler_name] = joblib.load(f"{model_dir}/{file}")
        
        self.is_trained = True
        logger.info("Models loaded successfully")
