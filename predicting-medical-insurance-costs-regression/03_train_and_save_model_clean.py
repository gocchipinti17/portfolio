import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import numpy as np
from data_cleaning import prep_data

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

# ======= SAVE MODEL =======
with open("final_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("💾 Model saved as 'final_model.pkl'")

