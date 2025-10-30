"""
Train SBA Expected Loss Models

This script trains the PD and LGD models and saves them for use.
"""

import pickle
from sklearn.model_selection import train_test_split
from features import prepare_data, create_preprocessor, transform_data
from models import train_pd_model, train_lgd_model


def main():
    print("="*60)
    print("TRAINING SBA EXPECTED LOSS MODEL")
    print("="*60)
    
    # Load data
    print("\n[1/5] Loading data...")
    X, y_pd, y_loss, df = prepare_data('SBAcase.csv')
    print(f"Loaded {len(X)} loans")
    print(f"Default rate: {y_pd.mean()*100:.2f}%")
    
    # Split data
    print("\n[2/5] Splitting data...")
    X_train, X_test, y_pd_train, y_pd_test, y_loss_train, y_loss_test = train_test_split(
        X, y_pd, y_loss, test_size=0.20, random_state=42, stratify=y_pd
    )
    
    # Preprocess
    print("\n[3/5] Preprocessing...")
    preprocessor = create_preprocessor()
    preprocessor.fit(X_train)
    X_train_t = transform_data(preprocessor, X_train)
    
    # Train PD model
    print("\n[4/5] Training PD model...")
    pd_model = train_pd_model(X_train_t, y_pd_train)
    print("✓ PD model trained")
    
    # Train LGD model (only on defaults)
    print("\n[5/5] Training LGD model...")
    defaults_mask = y_pd_train == 1
    X_defaults = X_train_t[defaults_mask]
    y_loss_defaults = y_loss_train[defaults_mask]
    lgd_model = train_lgd_model(X_defaults, y_loss_defaults)
    print("✓ LGD model trained")
    
    # Calculate calibration factor
    pd_pred = pd_model.predict_proba(X_train_t)[:, 1]
    lgd_pred = lgd_model.predict(X_train_t)
    el_pred = pd_pred * lgd_pred
    calibration_factor = y_loss_train.sum() / el_pred.sum()
    print(f"\nCalibration factor: {calibration_factor:.4f}")
    
    # Save models
    print("\nSaving models...")
    artifacts = {
        'preprocessor': preprocessor,
        'pd_model': pd_model,
        'lgd_model': lgd_model,
        'calibration_factor': calibration_factor
    }
    
    with open('sba_model.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("Models saved to: sba_model.pkl")
    print("="*60)


if __name__ == "__main__":
    main()
