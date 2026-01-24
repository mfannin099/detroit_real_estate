import numpy as np
from sklearn.linear_model import LinearRegression as SKLearnLR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from ..utils import DataCleaner

df = (DataCleaner("../data/airroi_listings.parquet")
      .load_data()
      .drop_columns()
      .clean_columns()
      .get_final_df())

print(df.head(3))