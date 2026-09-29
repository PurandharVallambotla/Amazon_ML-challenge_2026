"""XGBoost model training, persistence, and inference."""

import os
import xgboost as xgb
import numpy as np
from src.config import MODEL_PATH

def create_model():
    """Create tuned XGBoost classifier for entity matching."""
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric='logloss',
        random_state=42
    )

def train_and_save_model(X_train: np.ndarray, y_train: np.ndarray, save_path: str = MODEL_PATH):
    """Train XGBoost model and save to JSON format."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    clf = create_model()
    clf.fit(X_train, y_train)
    clf.save_model(save_path)
    print(f"Model saved to {save_path}")
    return clf

def load_trained_model(model_path: str = MODEL_PATH):
    """Load trained XGBoost model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Train the model first.")
    clf = create_model()
    clf.load_model(model_path)
    return clf

def predict_match_probabilities(clf, X_features: np.ndarray) -> np.ndarray:
    """Predict positive match probabilities for candidate pairs."""
    if len(X_features) == 0:
        return np.array([], dtype=np.float32)
    return clf.predict_proba(X_features)[:, 1]
