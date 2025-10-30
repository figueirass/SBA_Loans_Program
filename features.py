"""
Feature Engineering for SBA Expected Loss Model
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def prepare_data(filepath):
    """Load and prepare data with all features"""
    df = pd.read_csv(filepath)
    
    # Target variables
    df['ChgOffPrinGr'] = df['ChgOffPrinGr'].fillna(0).astype(float)
    y_loss = df['ChgOffPrinGr']
    y_pd = (y_loss > 0).astype(int)
    
    # Clean base features
    df['GrAppv'] = df['GrAppv'].fillna(0.0)
    df['SBA_Appv'] = df['SBA_Appv'].fillna(0.0)
    df['Term'] = df['Term'].fillna(0).astype(int)
    df['NoEmp'] = df['NoEmp'].fillna(0).astype(int)
    df['NewExist'] = df['NewExist'].fillna(0.0)
    df['IsNewBusiness'] = (df['NewExist'] == 1.0).astype(int)
    df['NAICS'] = df['NAICS'].astype(str).str[:2].replace({'0':'00'})
    
    # Engineer features
    df['SBA_Portion'] = np.where(df['GrAppv']>0, df['SBA_Appv']/df['GrAppv'], 0.0)
    df['Loan_per_Employee'] = np.where((df['NoEmp']+1)>0, df['GrAppv']/(df['NoEmp']+1), 0.0)
    df['Term_Years'] = df['Term'] / 12.0
    df['Debt_to_SBA'] = df['GrAppv'] - df['SBA_Appv']
    df['Log_GrAppv'] = np.log1p(df['GrAppv'])
    
    # Select features
    num_feats = ['GrAppv','SBA_Appv','Debt_to_SBA','Log_GrAppv','Term','Term_Years',
                 'NoEmp','IsNewBusiness','SBA_Portion','Loan_per_Employee']
    cat_feats = ['NAICS','State'] if 'State' in df.columns else ['NAICS']
    
    X = df[num_feats + cat_feats]
    
    return X, y_pd, y_loss, df


def create_preprocessor():
    """Create preprocessing pipeline"""
    num_feats = ['GrAppv','SBA_Appv','Debt_to_SBA','Log_GrAppv','Term','Term_Years',
                 'NoEmp','IsNewBusiness','SBA_Portion','Loan_per_Employee']
    cat_feats = ['NAICS','State']
    
    try:
        cat_tf = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        cat_tf = OneHotEncoder(handle_unknown="ignore", sparse=False)
    
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_feats),
        ("cat", cat_tf, cat_feats),
    ], remainder="drop")
    
    return preprocessor


def transform_data(preprocessor, X):
    """Transform data preserving DataFrame structure"""
    X_mat = preprocessor.transform(X)
    if hasattr(X_mat, "toarray"):
        X_mat = X_mat.toarray()
    
    try:
        cols = preprocessor.get_feature_names_out()
    except:
        cols = [f"f{i}" for i in range(X_mat.shape[1])]
    
    return pd.DataFrame(X_mat, index=X.index, columns=cols)
