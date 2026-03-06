"""
Step 3: Data Preparation & Transformation

- One-hot encode categorical variables
- Scale numeric features (for linear models only)
- Log-transform skewed target variable (charges)
- Create interaction terms for deeper model insight

Note on Missing Values & Random Forests:
Even though Random Forests are theoretically more robust to missing values
(e.g., using surrogate splits in some R packages), scikit-learn’s RandomForest
does NOT support missing values natively. All missing values must be handled
before modeling—even for tree-based models. We confirmed there are no missing
values in our dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def prep_data(df, scale=True, log_target=True, add_interactions=True):
    df = df.copy()

    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

    # Scale numeric variables (for linear regression)
    if scale:
        scaler = StandardScaler()
        for col in ['age', 'bmi', 'children']:
            if col in df.columns:
                df[col] = scaler.fit_transform(df[[col]])

    # Log-transform charges
    if log_target and 'charges' in df.columns:
        df['log_charges'] = np.log(df['charges'])

    # Add interaction terms
    if add_interactions:
        if 'age' in df.columns and 'bmi' in df.columns:
            df['age_bmi'] = df['age'] * df['bmi']
        if 'smoker_yes' in df.columns and 'age' in df.columns:
            df['smoker_age'] = df['smoker_yes'] * df['age']
        if 'region_northwest' in df.columns and 'bmi' in df.columns:
            df['bmi_region_nw'] = df['region_northwest'] * df['bmi']

    return df

# Load feather files
train_df = pd.read_feather("HW3/df_train.feather")
test_df = pd.read_feather("HW3/df_test.feather")

# Apply function
train_cleaned = prep_data(train_df)
test_cleaned = prep_data(test_df)

# Save outputs
train_cleaned.to_feather("HW3/train_cleaned.feather")
test_cleaned.to_feather("HW3/test_cleaned.feather")

print("✅ Step 3 complete! Cleaned data saved to feather format.")



