"""
SBA Expected Loss Model - Complete Analysis with Visualizations

Run this script after training to generate all performance graphs
and analysis for your report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import mean_absolute_error, r2_score

from features import prepare_data, create_preprocessor, transform_data
from models import train_pd_model, train_lgd_model

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*60)
print("SBA EXPECTED LOSS MODEL - COMPLETE ANALYSIS")
print("="*60)

# Load data
print("\n[1/8] Loading data...")
X, y_pd, y_loss, df = prepare_data('SBAcase.csv')
print(f"Dataset: {len(df):,} loans")
print(f"Default rate: {y_pd.mean()*100:.2f}%")

# Split data
print("\n[2/8] Splitting data...")
X_train, X_test, y_pd_train, y_pd_test, y_loss_train, y_loss_test = train_test_split(
    X, y_pd, y_loss, test_size=0.20, random_state=42, stratify=y_pd
)

# Preprocess
print("\n[3/8] Preprocessing...")
preprocessor = create_preprocessor()
preprocessor.fit(X_train)
X_train_t = transform_data(preprocessor, X_train)
X_test_t = transform_data(preprocessor, X_test)

# Train models or load existing
print("\n[4/8] Training/loading models...")
try:
    with open('sba_model.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    pd_model = artifacts['pd_model']
    lgd_model = artifacts['lgd_model']
    calibration_factor = artifacts['calibration_factor']
    print("✓ Loaded existing models")
except:
    print("Training new models...")
    pd_model = train_pd_model(X_train_t, y_pd_train)
    
    train_defaults = y_pd_train == 1
    lgd_model = train_lgd_model(X_train_t[train_defaults], y_loss_train[train_defaults])
    
    pd_pred = pd_model.predict_proba(X_train_t)[:, 1]
    lgd_pred = lgd_model.predict(X_train_t)
    el_pred = pd_pred * lgd_pred
    calibration_factor = y_loss_train.sum() / el_pred.sum()
    print("✓ Models trained")

# Make predictions
print("\n[5/8] Making predictions...")
pd_test = pd_model.predict_proba(X_test_t)[:, 1]
lgd_test = lgd_model.predict(X_test_t)
el_test = pd_test * lgd_test * calibration_factor

# ============================================
# VISUALIZATIONS
# ============================================

print("\n[6/8] Creating visualizations...")

# Figure 1: Data Overview
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Default rate by loan size
df['amount_bucket'] = pd.cut(df['GrAppv'], 
                              bins=[0, 50000, 150000, 350000, float('inf')],
                              labels=['<$50K', '$50-150K', '$150-350K', '>$350K'])
default_by_amount = df.groupby('amount_bucket', observed=True).agg({
    'ChgOffPrinGr': lambda x: (x > 0).mean() * 100
})

axes[0, 0].bar(range(len(default_by_amount)), default_by_amount['ChgOffPrinGr'], 
               color='indianred', edgecolor='black')
axes[0, 0].set_xticks(range(len(default_by_amount)))
axes[0, 0].set_xticklabels(default_by_amount.index, rotation=45)
axes[0, 0].set_ylabel('Default Rate (%)')
axes[0, 0].set_title('Default Rate by Loan Amount', fontweight='bold')
axes[0, 0].grid(axis='y', alpha=0.3)

# New vs Existing
df['business_type'] = df['IsNewBusiness'].map({0: 'Existing', 1: 'New'})
business_stats = df.groupby('business_type', observed=True).agg({
    'ChgOffPrinGr': lambda x: (x > 0).mean() * 100
})

axes[0, 1].bar(range(len(business_stats)), business_stats['ChgOffPrinGr'],
               color=['green', 'red'], edgecolor='black')
axes[0, 1].set_xticks(range(len(business_stats)))
axes[0, 1].set_xticklabels(business_stats.index, rotation=0)
axes[0, 1].set_ylabel('Default Rate (%)')
axes[0, 1].set_title('New vs Existing Businesses', fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)

# Top risky industries
industry_stats = df.groupby('NAICS', observed=True).agg({
    'ChgOffPrinGr': [('count', 'size'), ('default_rate', lambda x: (x > 0).mean() * 100)]
})
industry_stats.columns = ['count', 'default_rate']
industry_stats = industry_stats[industry_stats['count'] >= 20]
top_industries = industry_stats.nlargest(8, 'default_rate')

axes[1, 0].barh(range(len(top_industries)), top_industries['default_rate'],
                color='coral', edgecolor='black')
axes[1, 0].set_yticks(range(len(top_industries)))
axes[1, 0].set_yticklabels(top_industries.index)
axes[1, 0].set_xlabel('Default Rate (%)')
axes[1, 0].set_title('Top 8 Riskiest Industries (NAICS)', fontweight='bold')
axes[1, 0].grid(axis='x', alpha=0.3)
axes[1, 0].invert_yaxis()

# Loan amount distribution
axes[1, 1].hist(df['GrAppv'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[1, 1].axvline(df['GrAppv'].mean(), color='red', linestyle='--', 
                   label=f"Mean: ${df['GrAppv'].mean():,.0f}")
axes[1, 1].set_xlabel('Loan Amount ($)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Distribution of Loan Amounts', fontweight='bold')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('01_data_overview.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_data_overview.png")
plt.close()

# Figure 2: PD Model Performance
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ROC Curve
fpr, tpr, _ = roc_curve(y_pd_test, pd_test)
auc = roc_auc_score(y_pd_test, pd_test)

axes[0, 0].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.4f}')
axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('ROC Curve - PD Model', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Confusion Matrix
cm = confusion_matrix(y_pd_test, (pd_test >= 0.5).astype(int))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1], square=True)
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')
axes[0, 1].set_title('Confusion Matrix', fontweight='bold')

# PD Distribution
axes[1, 0].hist(pd_test, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
axes[1, 0].axvline(pd_test.mean(), color='red', linestyle='--',
                   label=f'Mean: {pd_test.mean():.2%}')
axes[1, 0].set_xlabel('Predicted Probability of Default')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of PD Predictions', fontweight='bold')
axes[1, 0].legend()

# PD by actual outcome
axes[1, 1].hist(pd_test[y_pd_test == 0], bins=30, alpha=0.6, 
                label='No Default', color='green', edgecolor='black')
axes[1, 1].hist(pd_test[y_pd_test == 1], bins=30, alpha=0.6,
                label='Default', color='red', edgecolor='black')
axes[1, 1].set_xlabel('Predicted PD')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('PD by Actual Outcome', fontweight='bold')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('02_pd_model_performance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_pd_model_performance.png")
plt.close()

# Figure 3: LGD Model Performance
test_defaults = y_pd_test == 1
lgd_test_defaults = lgd_test[test_defaults]
y_loss_test_defaults = y_loss_test[test_defaults]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Predicted vs Actual
axes[0, 0].scatter(lgd_test_defaults, y_loss_test_defaults, alpha=0.5, s=30)
max_val = max(lgd_test_defaults.max(), y_loss_test_defaults.max())
axes[0, 0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect')
axes[0, 0].set_xlabel('Predicted Loss ($)')
axes[0, 0].set_ylabel('Actual Loss ($)')
axes[0, 0].set_title('Predicted vs Actual Loss', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Residuals
residuals = y_loss_test_defaults.values - lgd_test_defaults
axes[0, 1].scatter(lgd_test_defaults, residuals, alpha=0.5, s=30)
axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Predicted Loss ($)')
axes[0, 1].set_ylabel('Residual ($)')
axes[0, 1].set_title('Residual Plot', fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# Distribution
axes[1, 0].hist(y_loss_test_defaults, bins=50, alpha=0.6, 
                label='Actual', color='blue', edgecolor='black')
axes[1, 0].hist(lgd_test_defaults, bins=50, alpha=0.6,
                label='Predicted', color='orange', edgecolor='black')
axes[1, 0].set_xlabel('Loss Amount ($)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Actual vs Predicted Loss Distribution', fontweight='bold')
axes[1, 0].legend()

# Performance metrics
mae = mean_absolute_error(y_loss_test_defaults, lgd_test_defaults)
r2 = r2_score(y_loss_test_defaults, lgd_test_defaults)
metrics_text = f"MAE: ${mae:,.0f}\nR²: {r2:.4f}\nMean Actual: ${y_loss_test_defaults.mean():,.0f}\nMean Predicted: ${lgd_test_defaults.mean():,.0f}"
axes[1, 1].text(0.5, 0.5, metrics_text, fontsize=14, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1, 1].set_title('LGD Model Metrics', fontweight='bold')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('03_lgd_model_performance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_lgd_model_performance.png")
plt.close()

# Figure 4: Expected Loss Analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# EL Distribution
axes[0, 0].hist(el_test, bins=50, edgecolor='black', alpha=0.7, color='purple')
axes[0, 0].axvline(el_test.mean(), color='red', linestyle='--',
                   label=f'Mean: ${el_test.mean():,.0f}')
axes[0, 0].set_xlabel('Expected Loss ($)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Expected Loss', fontweight='bold')
axes[0, 0].legend()

# EL vs Actual Loss
axes[0, 1].scatter(el_test, y_loss_test, alpha=0.5, s=20)
max_val = max(el_test.max(), y_loss_test.max())
axes[0, 1].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect')
axes[0, 1].set_xlabel('Predicted EL ($)')
axes[0, 1].set_ylabel('Actual Loss ($)')
axes[0, 1].set_title('Predicted EL vs Actual Loss', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Cumulative EL
sorted_idx = np.argsort(el_test)[::-1]
cumsum_pred = np.cumsum(el_test[sorted_idx])
cumsum_actual = np.cumsum(y_loss_test.values[sorted_idx])
percentiles = np.arange(1, len(sorted_idx) + 1) / len(sorted_idx) * 100

axes[1, 0].plot(percentiles, cumsum_pred, label='Predicted EL', linewidth=2)
axes[1, 0].plot(percentiles, cumsum_actual, label='Actual Loss', linewidth=2)
axes[1, 0].set_xlabel('Percentile of Loans')
axes[1, 0].set_ylabel('Cumulative Loss ($)')
axes[1, 0].set_title('Cumulative Loss Distribution', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# EL by loan bucket
test_df = X_test.copy()
test_df['predicted_el'] = el_test
test_df['actual_loss'] = y_loss_test.values
test_df['bucket'] = pd.cut(test_df['GrAppv'],
                            bins=[0, 50000, 150000, 350000, float('inf')],
                            labels=['<$50K', '$50-150K', '$150-350K', '>$350K'])

bucket_stats = test_df.groupby('bucket', observed=True).agg({'predicted_el': 'sum', 'actual_loss': 'sum'})

x_pos = np.arange(len(bucket_stats))
width = 0.35
axes[1, 1].bar(x_pos - width/2, bucket_stats['predicted_el'], width, label='Predicted', color='orange')
axes[1, 1].bar(x_pos + width/2, bucket_stats['actual_loss'], width, label='Actual', color='blue')
axes[1, 1].set_xlabel('Loan Amount')
axes[1, 1].set_ylabel('Total Loss ($)')
axes[1, 1].set_title('Expected Loss by Loan Size', fontweight='bold')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(bucket_stats.index, rotation=45)
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('04_expected_loss_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_expected_loss_analysis.png")
plt.close()

# Figure 5: Risk Segmentation
test_df['risk_score'] = el_test
# Use duplicates='drop' to handle many zero/identical risk scores
try:
    test_df['risk_category'] = pd.qcut(test_df['risk_score'], q=5,
                                        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                                        duplicates='drop')
except ValueError:
    # If still can't create 5 categories, use 3 instead
    test_df['risk_category'] = pd.qcut(test_df['risk_score'], q=3,
                                        labels=['Low', 'Medium', 'High'],
                                        duplicates='drop')

risk_summary = test_df.groupby('risk_category', observed=True).agg({
    'GrAppv': 'count',
    'predicted_el': 'mean',
    'actual_loss': lambda x: (x > 0).mean()
})
risk_summary.columns = ['Count', 'Avg EL', 'Default Rate']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loan count by risk
risk_summary['Count'].plot(kind='bar', ax=axes[0, 0], color='steelblue', edgecolor='black')
axes[0, 0].set_title('Loan Count by Risk Category', fontweight='bold')
axes[0, 0].set_ylabel('Number of Loans')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(axis='y', alpha=0.3)

# Avg EL by risk
risk_summary['Avg EL'].plot(kind='bar', ax=axes[0, 1], color='coral', edgecolor='black')
axes[0, 1].set_title('Average Expected Loss by Risk', fontweight='bold')
axes[0, 1].set_ylabel('Avg EL ($)')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(axis='y', alpha=0.3)

# Default rate by risk
risk_summary['Default Rate'].plot(kind='bar', ax=axes[1, 0], color='indianred', edgecolor='black')
axes[1, 0].set_title('Default Rate by Risk Category', fontweight='bold')
axes[1, 0].set_ylabel('Default Rate')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(axis='y', alpha=0.3)

# Summary table
table_data = risk_summary.round(2).reset_index().values
axes[1, 1].axis('tight')
axes[1, 1].axis('off')
table = axes[1, 1].table(cellText=table_data, 
                         colLabels=['Risk Category', 'Count', 'Avg EL ($)', 'Default Rate'],
                         cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
axes[1, 1].set_title('Risk Segmentation Summary', fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('05_risk_segmentation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_risk_segmentation.png")
plt.close()

# Print summary
print("\n[7/8] Model Performance Summary")
print("="*60)
print(f"PD Model AUC: {auc:.4f}")
print(f"LGD Model MAE: ${mae:,.2f}")
print(f"LGD Model R²: {r2:.4f}")
print(f"Calibration Factor: {calibration_factor:.4f}")
print(f"Total Predicted EL: ${el_test.sum():,.2f}")
print(f"Total Actual Loss: ${y_loss_test.sum():,.2f}")
print(f"Prediction Ratio: {el_test.sum() / y_loss_test.sum():.4f}")

print("\n[8/8] Analysis complete!")
print("="*60)
print("\nGenerated files:")
print("  01_data_overview.png")
print("  02_pd_model_performance.png")
print("  03_lgd_model_performance.png")
print("  04_expected_loss_analysis.png")
print("  05_risk_segmentation.png")
print("\n✓ All visualizations saved!")