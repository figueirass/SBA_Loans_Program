"""
SBA Loan Guarantee Fee Calculator

Run this script to get a quote for an SBA loan guarantee.
It will automatically train models if they don't exist.
"""

import pickle
import os
import numpy as np
import pandas as pd
from features import create_preprocessor, transform_data


def calculate_sba_guarantee(approved_amount):
    """Calculate SBA guarantee amount based on loan size"""
    if approved_amount <= 150000:
        return approved_amount * 0.85
    else:
        return approved_amount * 0.75


def calculate_monthly_payment(principal, annual_rate, term_months):
    """Calculate monthly loan payment"""
    if annual_rate == 0:
        return principal / term_months
    
    monthly_rate = (annual_rate / 100) / 12
    payment = principal * (monthly_rate * (1 + monthly_rate)**term_months) / \
              ((1 + monthly_rate)**term_months - 1)
    return payment


def create_loan_features(approved_amount, term_months, num_employees, 
                         is_new_business, naics_code, state_code):
    """Create feature vector for a loan"""
    sba_guaranteed = calculate_sba_guarantee(approved_amount)
    
    features = {
        'GrAppv': approved_amount,
        'SBA_Appv': sba_guaranteed,
        'Term': term_months,
        'NoEmp': num_employees,
        'IsNewBusiness': 1 if is_new_business else 0,
        'NewExist': 1.0 if is_new_business else 2.0,
        'NAICS': naics_code,
        'State': state_code,
        'SBA_Portion': sba_guaranteed / approved_amount if approved_amount > 0 else 0,
        'Loan_per_Employee': approved_amount / (num_employees + 1),
        'Term_Years': term_months / 12.0,
        'Debt_to_SBA': approved_amount - sba_guaranteed,
        'Log_GrAppv': np.log1p(approved_amount)
    }
    
    return pd.DataFrame([features])


def load_models():
    """Load trained models or train if they don't exist"""
    if not os.path.exists('sba_model.pkl'):
        print("Models not found. Training now...")
        print("This will take a few minutes...\n")
        import train
        train.main()
        print("\n")
    
    with open('sba_model.pkl', 'rb') as f:
        return pickle.load(f)


def calculate_quote(approved_amount, term_months, num_employees, 
                   is_new_business, naics_code, state_code, bank_rate):
    """Calculate complete loan quote"""
    
    # Load models
    artifacts = load_models()
    
    # Create features
    loan_df = create_loan_features(
        approved_amount, term_months, num_employees,
        is_new_business, naics_code, state_code
    )
    
    # Preprocess
    X_processed = transform_data(artifacts['preprocessor'], loan_df)
    
    # Predict
    pd_pred = artifacts['pd_model'].predict_proba(X_processed)[:, 1][0]
    lgd_pred = artifacts['lgd_model'].predict(X_processed)[0]
    el_pred = pd_pred * lgd_pred * artifacts['calibration_factor']
    
    # Calculate fee (EL + 20% margin)
    sba_guaranteed = calculate_sba_guarantee(approved_amount)
    guarantee_fee = min(el_pred * 1.20, sba_guaranteed)
    guarantee_fee = max(guarantee_fee, sba_guaranteed * 0.005)  # Min 0.5%
    
    # Calculate payment
    total_financed = approved_amount + guarantee_fee
    monthly_payment = calculate_monthly_payment(total_financed, bank_rate, term_months)
    
    return {
        'approved_amount': approved_amount,
        'sba_guaranteed': sba_guaranteed,
        'pd': pd_pred,
        'lgd': lgd_pred,
        'expected_loss': el_pred,
        'guarantee_fee': guarantee_fee,
        'total_financed': total_financed,
        'monthly_payment': monthly_payment,
        'term_months': term_months,
        'bank_rate': bank_rate
    }


def print_quote(quote):
    """Print formatted quote"""
    print("\n" + "="*60)
    print("SBA LOAN GUARANTEE QUOTE")
    print("="*60)
    
    print("\n--- Risk Assessment ---")
    print(f"Loan Amount:              ${quote['approved_amount']:,.2f}")
    print(f"SBA Guaranteed Amount:    ${quote['sba_guaranteed']:,.2f}")
    print(f"Probability of Default:   {quote['pd']*100:.2f}%")
    print(f"Loss Given Default:       ${quote['lgd']:,.2f}")
    print(f"Expected Loss:            ${quote['expected_loss']:,.2f}")
    
    print("\n--- SBA Guarantee Fee ---")
    print(f"Upfront Fee:              ${quote['guarantee_fee']:,.2f}")
    
    print("\n--- Monthly Payment ---")
    print(f"Loan Amount:              ${quote['approved_amount']:,.2f}")
    print(f"+ Guarantee Fee:          ${quote['guarantee_fee']:,.2f}")
    print(f"= Total Financed:         ${quote['total_financed']:,.2f}")
    print(f"\nInterest Rate:            {quote['bank_rate']:.2f}%")
    print(f"Term:                     {quote['term_months']} months")
    print(f"\nMONTHLY PAYMENT:          ${quote['monthly_payment']:,.2f}")
    print("="*60 + "\n")


def main():
    """Main interactive quoter"""
    print("\n" + "="*60)
    print("SBA LOAN GUARANTEE CALCULATOR")
    print("="*60)
    print("\nEnter loan details:\n")
    
    try:
        approved_amount = float(input("1. Loan amount ($): "))
        term_months = int(input("2. Term (months): "))
        num_employees = int(input("3. Number of employees: "))
        is_new = input("4. New business? (y/n): ").lower() == 'y'
        naics = input("5. NAICS code (2 digits, e.g., 72): ").strip()
        state = input("6. State (e.g., CA): ").strip().upper()
        bank_rate = float(input("7. Bank interest rate (%): "))
        
        print("\nCalculating quote...")
        
        quote = calculate_quote(
            approved_amount, term_months, num_employees,
            is_new, naics, state, bank_rate
        )
        
        print_quote(quote)
        
    except ValueError:
        print("\n❌ Invalid input. Please enter valid numbers.")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
