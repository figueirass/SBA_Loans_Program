# SBA Expected Loss Model

A machine learning system for estimating Expected Loss (EL) on Small Business Administration (SBA) loan guarantees, designed from the guarantor's perspective rather than the lender's.

## 🎯 Project Overview

### The SBA Perspective vs. Bank Perspective

This project models credit risk from the **SBA's viewpoint as a guarantor**, which is fundamentally different from traditional bank credit models:

| Aspect | Bank Model | SBA Model (This Project) |
|--------|-----------|--------------------------|
| **Question** | Should we approve this loan? (Yes/No) | How much money will we lose on this guarantee? ($) |
| **Model Type** | Classification (PD model) | Regression (EL model) |
| **Output** | Probability of Default | Expected Loss in Dollars |
| **Decision** | Approve/Reject | Guarantee Fee Calculation |

### Key Concept: Expected Loss

The SBA needs to estimate **Expected Loss (EL)** for each guaranteed loan:

```
Expected Loss (EL) = Probability of Default (PD) × Loss Given Default (LGD)
```

- **PD**: Probability that the borrower will default (0 to 1)
- **LGD**: Dollar amount the SBA will lose if default occurs
- **EL**: Expected dollar amount the SBA will lose on this guarantee

This EL estimate drives the **guarantee fee** (upfront fee) that the SBA charges.

## 🏗️ Architecture

The model uses a **two-stage approach**:

### Stage 1: Probability of Default (PD) Model
- **Type**: Binary Classification
- **Models**: Logistic Regression, Random Forest
- **Output**: Probability that loan will default
- **Calibration**: Isotonic regression for probability accuracy

### Stage 2: Loss Given Default (LGD) Model
- **Type**: Regression (trained only on defaulted loans)
- **Models**: Random Forest, Gradient Boosting, XGBoost
- **Output**: Dollar amount of loss if default occurs

### Stage 3: Expected Loss Calculation
- **Formula**: EL = PD × LGD
- **Calibration**: Global and segment-specific adjustments
- **Segments**: Loan amount, term, industry (NAICS)

## 📁 Project Structure

```
sba-expected-loss-model/
├── README.md
├── requirements.txt
├── train_model.py          # Main training pipeline
├── predict_loan.py         # Interactive quote calculator
│
├── data/
│   └── SBAcase.csv         # Historical SBA loan data
│
├── models/
│   └── sba_expected_loss_model.pkl  # Trained models (generated)
│
├── src/
│   ├── preprocessing/
│   │   └── feature_engineering.py   # Data preparation & feature engineering
│   │
│   ├── models/
│   │   ├── pd_model.py              # Probability of Default models
│   │   ├── lgd_model.py             # Loss Given Default models
│   │   └── expected_loss.py         # EL calculation & calibration
│   │
│   └── utils/
│       └── loan_calculator.py       # Guarantee fee calculator
│
└── notebooks/
    └── SBA_Project.ipynb            # Original exploration notebook
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sba-expected-loss-model.git
cd sba-expected-loss-model

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place your SBA loan data in the `data/` directory:
```bash
data/SBAcase.csv
```

### 3. Train Models

```bash
python train_model.py
```

This will:
- Load and preprocess the data
- Train multiple PD and LGD models
- Select the best models based on validation performance
- Calculate calibration factors
- Save all artifacts to `models/sba_expected_loss_model.pkl`

**Expected Output:**
```
==========================================
SBA EXPECTED LOSS MODEL - TRAINING PIPELINE
==========================================

[1/6] Loading and preparing data...
Dataset: 5000 loans
Default rate: 15.23%
Average loss amount: $45,678.00

...

Best PD Model: Random Forest with validation AUC = 0.7856
Best LGD Model: XGBoost with validation MAE = $12,345.67

✓ All models ready for deployment
```

### 4. Generate Loan Quotes

```bash
python predict_loan.py
```

Interactive example:
```
SBA LOAN GUARANTEE QUOTE CALCULATOR
====================================

Please enter the loan details:

1. Bank-approved loan amount (e.g., 150000): $200000
2. Loan term in MONTHS (e.g., 120): 120
3. Number of employees (e.g., 8): 15
4. Is this a new business? (y/n): n
5. NAICS code (first 2 digits, e.g., 72 for restaurant): 72
6. State code (e.g., CA): CA
7. Bank's annual interest rate % (e.g., 6.5): 6.5

=====================================
SBA LOAN GUARANTEE QUOTE
=====================================

--- Risk Assessment ---
Approved Loan Amount:        $200,000.00
SBA Guaranteed Amount:       $150,000.00
Probability of Default:      18.45%
Loss Given Default:          $62,341.23
Expected Loss (Calibrated):  $13,806.35

--- Guarantee Fee ---
Upfront Guarantee Fee:       $16,567.62

--- Loan Payment ---
Loan Amount Requested:       $200,000.00
Guarantee Fee (Financed):  + $16,567.62
TOTAL AMOUNT FINANCED:       $216,567.62

Interest Rate:               6.50%
Loan Term:                   120 months

MONTHLY PAYMENT:             $2,458.91
=====================================
```

## 📊 Features

### Engineered Features

The model uses domain-specific features designed for SBA loan risk assessment:

**Numerical Features:**
- `GrAppv`: Gross approved amount
- `SBA_Appv`: SBA guaranteed amount
- `SBA_Portion`: Guarantee coverage ratio (SBA_Appv / GrAppv)
- `Loan_per_Employee`: Loan size relative to business scale
- `Term_Years`: Loan term in years
- `Debt_to_SBA`: Bank's exposure (non-guaranteed portion)
- `Log_GrAppv`: Log-transformed loan amount
- `IsNewBusiness`: New vs. existing business

**Categorical Features:**
- `NAICS`: Industry classification (2-digit codes)
- `State`: Geographic location

## 🎓 Model Details

### PD Model Selection Criteria
- **Metric**: ROC AUC
- **Calibration**: Isotonic regression
- **Typical Performance**: AUC ~0.75-0.80

### LGD Model Selection Criteria
- **Metric**: Mean Absolute Error (MAE)
- **Training Set**: Only defaulted loans (PD = 1)
- **Typical Performance**: MAE ~$10,000-$15,000

### Calibration Strategy

1. **Global Calibration**: Adjusts overall prediction level
   ```
   Calibration Factor = Total Actual Losses / Total Predicted Losses
   ```

2. **Segmented Calibration**: Fine-tunes by loan characteristics
   - Loan amount buckets: <$50K, $50-150K, $150-350K, >$350K
   - Term buckets: <5yr, 5-10yr, 10-20yr, >20yr
   - Industry sectors: By NAICS code

## 📈 Performance Metrics

### Model Evaluation

**PD Model:**
- ROC AUC (Test): Measures discrimination ability
- Calibration Plot: Predicted vs. actual default rates
- Confusion Matrix: Classification accuracy

**LGD Model:**
- MAE (Test): Average dollar prediction error
- RMSE: Root mean squared error
- R²: Variance explained

**Overall EL Model:**
- Total predicted vs. actual losses
- Segment-specific accuracy
- Guarantee fee profitability analysis

## 🔧 Customization

### Adjusting Guarantee Fee Margin

In `src/utils/loan_calculator.py`, modify the margin:

```python
def calculate_guarantee_fee(expected_loss, sba_guaranteed_amount, margin=0.20):
    # margin = 0.20 means 20% above expected loss
    fee = expected_loss * (1 + margin)
    ...
```

### Adding New Features

1. Add feature engineering in `src/preprocessing/feature_engineering.py`
2. Update feature lists in `get_feature_lists()`
3. Retrain models with `python train_model.py`

### Trying Different Models

Modify model parameters in training functions:
- `src/models/pd_model.py` - PD model configurations
- `src/models/lgd_model.py` - LGD model configurations

## 📚 Background: SBA Loan Guarantee Program

The Small Business Administration (SBA) provides loan guarantees to reduce risk for lenders:

1. **Bank approves** a loan to a small business
2. **SBA guarantees** 75-85% of the loan amount
3. **Bank charges** interest to the borrower
4. **SBA charges** an upfront guarantee fee
5. If the loan defaults:
   - Bank tries to recover funds
   - **SBA pays** the guaranteed portion of net loss
   - This is the **actual loss** the SBA incurs

### Why This Model Matters

- **Risk Management**: SBA needs to understand expected losses
- **Fee Setting**: Guarantee fees should cover expected losses + margin
- **Program Sustainability**: Proper pricing ensures long-term viability
- **Portfolio Analysis**: Identify high-risk segments

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add feature importance analysis
- [ ] Implement SHAP values for interpretability
- [ ] Create API endpoint for predictions
- [ ] Add unit tests
- [ ] Create visualization dashboard
- [ ] Implement time-series cross-validation
- [ ] Add economic scenario analysis

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or collaboration:
- **Project Lead**: [Your Name]
- **Email**: [your.email@example.com]
- **LinkedIn**: [Your LinkedIn Profile]

## 🙏 Acknowledgments

- SBA for historical loan data
- Credit risk modeling best practices from Basel III framework
- Open source machine learning community

---

**Note**: This model is for educational and analytical purposes. Actual SBA guarantee fees involve additional factors including regulatory requirements, program policies, and economic conditions.
