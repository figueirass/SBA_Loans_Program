"""
PD and LGD Models for SBA Expected Loss
"""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBRegressor


def train_pd_model(X_train, y_train):
    """Train Probability of Default model"""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=50,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Calibrate
    calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    calibrated.fit(X_train, y_train)
    
    return calibrated


def train_lgd_model(X_train, y_train):
    """Train Loss Given Default model (only on defaults)"""
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def calculate_expected_loss(pd_pred, lgd_pred, calibration_factor=1.0):
    """Calculate Expected Loss: EL = PD × LGD"""
    return pd_pred * lgd_pred * calibration_factor
