"""
04_predict_from_excel.py

Author: Kache Lee

Updated to match the latest version of prep_data() which includes interaction terms.

This script:
- Loads Excel (CSV) input
- Applies median imputation and categorical encoding
- Adds interaction terms
- Aligns with model expectations
- Predicts medical insurance charges using a pre-trained XGBoost booster
"""

import pandas as pd
import xgboost as xgb
import pickle
from typing import Optional
from data_cleaning import prep_data

from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)

from typing import Optional

def load_booster(path: str = 'final_model.json') -> Optional[xgb.Booster]:
    """Load the XGBoost booster model from a JSON file."""
    try:
        booster = xgb.Booster()
        booster.load_model(path)
        return booster
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def predict_from_excel(file_path: str, impute_values: dict) -> None:
    """Clean input, align columns, predict charges, and export result."""
    print("🚀 Starting prediction...")

    # Step 1: Load the CSV input
    df = pd.read_csv(file_path)

    # ✅ Step 1.5: Remove any unnamed columns (like 'Unnamed: 6' from Excel)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Step 2: Apply cleaning function (includes encoding + interaction terms)
    df_cleaned, _ = prep_data(df, impute_values=impute_values)

    # Step 3: Add missing dummy columns (in case input is missing regions)
    expected_cols = [
        'region_northwest', 'region_southeast', 'region_southwest',
        'bmi_x_smoker', 'bmi_x_sex'
    ]
    for col in expected_cols:
        if col not in df_cleaned.columns:
            df_cleaned[col] = 0

    # Step 4: Load booster model
    booster = load_booster()
    print("✅ Booster loaded")

    if booster:
        # Step 5: Align columns with booster expectations
        df_cleaned = df_cleaned.loc[:, booster.feature_names]
        print("✅ Columns aligned")

        # Step 6: Convert to DMatrix and predict
        dmatrix = xgb.DMatrix(df_cleaned)
        print("✅ DMatrix created")

        prediction = booster.predict(dmatrix)
        print(f"\n💡 Predicted Charges: ${prediction[0]:,.2f}")

        # Step 7: Save output
        output_df = df.copy()
        output_df['predicted_charges'] = prediction[0]
        output_df.to_csv("excel_test_row_with_prediction.csv", index=False)
        print("📁 Saved predicted output to 'excel_test_row_with_prediction.csv'")
    else:
        print("❌ Booster is None — model could not be loaded.")


if __name__ == "__main__":
    impute_values = {
        'age': 39.5,
        'bmi': 30.21,
        'children': 1.0
    }
    predict_from_excel("excel_test_row.csv", impute_values)
