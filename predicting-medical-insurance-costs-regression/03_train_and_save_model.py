"""
03_train_and_save_model.py

Author: Reuben Vincent

This script:
- Loads and cleans training data using `prep_data`
- Trains an XGBoost regressor with GridSearchCV
- Evaluates RMSE and MAE
- Saves the trained model safely using `joblib.dump`
"""

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_cleaning import prep_data

from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)

# ======= LOAD + CLEAN RAW TRAINING DATA =======
raw_df = pd.read_feather("df_train.feather")
train_df, _ = prep_data(raw_df)

# ======= SPLIT X AND y =======
X = train_df.drop(columns=["charges"])
y = train_df["charges"]

# ======= DEFINE GRID + MODEL =======
param_grid = {
    "n_estimators": [100],
    "max_depth": [3, 5],
    "learning_rate": [0.1]
}

xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

grid_search = GridSearchCV(
    xgb,
    param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

# ======= TRAIN MODEL =======
grid_search.fit(X, y)

# ======= EVALUATE + PRINT =======
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)

print("✅ Best Parameters:", grid_search.best_params_)
print(f"📉 RMSE: {rmse:.2f}")
print(f"📊 MAE: {mae:.2f}")

# ======= SAVE MODEL SAFELY IN XGBoost JSON FORMAT =======
booster = best_model.get_booster()
booster.save_model("final_model.json")
print("💾 Booster saved as 'final_model.json'")
