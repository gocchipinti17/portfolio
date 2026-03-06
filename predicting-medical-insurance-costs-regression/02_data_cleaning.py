import pandas as pd

def prep_data(df, impute_values=None):
    """
    Author: Kache Lee and Gina Occhipinti
    02_data_cleaning.py

    This script includes the prep_data() function used across all stages of the pipeline.
    Although the original dataset had no missing values (confirmed during EDA), we included
    median imputation logic to prepare for real-world use cases and to meet feedback from
    Professor Veliche. The function fills missing values in age, bmi, and children using
    medians calculated during training, and applies the same values to new input rows
    (e.g., Excel input) during prediction.

    prep_data() also encodes categorical variables and adds interaction terms flagged during EDA.

    Universal data cleaning function used across all scripts:
    - Encodes binary and categorical features
    - Handles missing value imputation using median strategy
    - Adds interaction terms: BMI × Smoker, BMI × Sex
    - Returns a cleaned DataFrame and a dictionary of imputed values

    Used by:
    - Gina (cleaning + feather export)
    - Reuben (cleaned training data)
    - Kache (Excel prediction cleaning)

    Parameters:
    - df: input DataFrame (raw or 1-row Excel)
    - impute_values: dict (optional); if None, calculates medians for reuse
    """

    df = df.copy()

    # Binary encoding
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})

    # One-hot encoding for region
    df = pd.get_dummies(df, columns=['region'], drop_first=True)

    # Define numeric columns to check for missing values
    numeric_cols = ['age', 'bmi', 'children']

    # If no impute_values passed in, calculate medians (training)
    if impute_values is None:
        impute_values = {col: df[col].median() for col in numeric_cols}

    # Fill missing values with imputation values
    for col in numeric_cols:
        df[col] = df[col].fillna(impute_values[col])

    # Interaction terms
    df['bmi_x_smoker'] = df['bmi'] * df['smoker']
    df['bmi_x_sex'] = df['bmi'] * df['sex']

    return df, impute_values

# ================================
# Test Block
# ================================
if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path
    import os

    # Automatically set working directory to script’s location
    os.chdir(Path(__file__).resolve().parent)

    # Load the feather file
    df = pd.read_feather("df_train.feather")

    # Visibly show missing values just for demo clarity
    print("\n🔍 Any missing values in the training data?")
    print(df[['age', 'bmi', 'children']].isnull().sum())

    # Run the cleaning function
    cleaned_df, impute_vals = prep_data(df)

    # Show result
    print("\n✅ Cleaned Data (first 5 rows):")
    print(cleaned_df.head())

    print("\n🧠 Imputation Values:")
    print(impute_vals)