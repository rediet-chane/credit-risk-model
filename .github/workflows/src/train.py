"""
Model Training and Tracking with MLflow
Task 5: Model Training and Experiment Tracking
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')


class CreditRiskModelTrainer:
    """
    Model training pipeline with MLflow tracking
    Task 5 Components: train/test split, hyperparameter tuning, MLflow tracking, model registration
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            'LogisticRegression': LogisticRegression(random_state=random_state, max_iter=1000),
            'DecisionTree': DecisionTreeClassifier(random_state=random_state),
            'RandomForest': RandomForestClassifier(random_state=random_state, n_jobs=-1)
        }
        
        self.param_grids = {
            'LogisticRegression': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l2']
            },
            'DecisionTree': {
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10]
            },
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5]
            }
        }
        
        self.best_model = None
        self.best_model_name = None
        self.best_params = None
        self.best_metrics = {}
        
    def prepare_data(self, features_df, target_series):
        """
        Split data into train and test sets
        Task 5 Component 2: Train/test split with random_state
        """
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, target_series,
            test_size=0.2,
            random_state=self.random_state,
            stratify=target_series
        )
        return X_train, X_test, y_train, y_test
    
    def train_and_track(self, X_train, X_test, y_train, y_test):
        """
        Train all models, track experiments with MLflow
        Task 5 Components 3, 4, 5: Hyperparameter tuning, MLflow tracking, model registration
        """
        results = []
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            
            # Get param grid for this model
            param_grid = self.param_grids.get(model_name, {})
            
            if param_grid:
                # Hyperparameter tuning with GridSearchCV
                grid_search = GridSearchCV(
                    model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
            else:
                best_model = model
                best_params = {}
                best_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            # Log with MLflow
            with mlflow.start_run(run_name=model_name):
                mlflow.log_params(best_params)
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(best_model, model_name)
                
                # Register the best model (for RandomForest if ROC-AUC > 0.8)
                if model_name == 'RandomForest' and metrics['roc_auc'] > 0.8:
                    mlflow.register_model(
                        f"runs:/{mlflow.active_run().info.run_id}/{model_name}",
                        "CreditRiskModel"
                    )
            
            results.append({
                'model': model_name,
                'best_params': best_params,
                'metrics': metrics
            })
            
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
        
        # Select best model (highest ROC-AUC)
        best_result = max(results, key=lambda x: x['metrics']['roc_auc'])
        self.best_model_name = best_result['model']
        self.best_params = best_result['best_params']
        self.best_metrics = best_result['metrics']
        
        # Retrain best model on full training data
        if self.best_model_name == 'LogisticRegression':
            self.best_model = LogisticRegression(**self.best_params, random_state=self.random_state, max_iter=1000)
        elif self.best_model_name == 'DecisionTree':
            self.best_model = DecisionTreeClassifier(**self.best_params, random_state=self.random_state)
        else:
            self.best_model = RandomForestClassifier(**self.best_params, random_state=self.random_state, n_jobs=-1)
        
        self.best_model.fit(X_train, y_train)
        
        return results
    
    def get_best_model(self):
        """Return the best trained model"""
        return self.best_model, self.best_model_name, self.best_metrics


if __name__ == "__main__":
    print("Model Training Module Loaded")
    print("Components included:")
    print("  - Train/test split with random_state")
    print("  - Multiple models (LogisticRegression, DecisionTree, RandomForest)")
    print("  - GridSearchCV hyperparameter tuning")
    print("  - MLflow tracking (params, metrics, artifacts)")
    print("  - Model registration in MLflow Registry")
    print("  - Evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC")