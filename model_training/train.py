import numpy as np
from sklearn.linear_model import LinearRegression as SKLearnLR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from utils import DataCleaner
data_path = os.path.join(parent_dir, "data", "airroi_listings.parquet")

df = (DataCleaner(data_path)
      .load_data()
      .drop_columns()
      .drop_columns_additional_ml_cols()
      .clean_columns()
      #.train_test(target_col='ttm_revenue', )
      .get_final_df()
      )

print(df.head(3))
print(df.columns)

nan_counts = df.isnull().sum()
print(nan_counts)