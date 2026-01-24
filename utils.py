import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, path):
        self.path = path
        self.df = None
        self.target_col = None
    
    def load_data(self):
        self.df = pd.read_parquet(self.path)
        return self

    def drop_columns(self):
        cols_to_drop = ["cover_photo_url", "listing_id", "host_id", "host_name",
                        "cohost_ids", "cohost_names", 'registration', "instant_book",
                        "ttm_revenue_native", "ttm_blocked_days",  'ttm_revpar', 'ttm_revpar_native', 'ttm_adjusted_revpar',
                        'ttm_avg_rate_native', 'ttm_occupancy', 'ttm_adjusted_occupancy',
                        'ttm_revpar', 'ttm_revpar_native', 'ttm_adjusted_revpar',
                        'ttm_adjusted_revpar_native', 'ttm_blocked_days'
                        'l90d_revenue', 'l90d_revenue_native', 'l90d_avg_rate', 'l90d_avg_rate_native',
                        'l90d_occupancy', 'l90d_adjusted_occupancy', 'l90d_revpar',
                        'l90d_revpar_native', 'l90d_adjusted_revpar',
                        'l90d_adjusted_revpar_native', 'l90d_reserved_days',
                        'l90d_blocked_days', 'l90d_available_days', 'l90d_total_days' ]

        self.df = self.df.drop(columns=cols_to_drop, errors='ignore')
        return self

    def drop_columns_additional_ml_cols(self):
        cols_to_drop = ["listing_name", "latitude", "longitude", "extra_guest_fee",
        "amenities", 'cancellation_policy', 'currency', "ttm_avg_rate", 
        "ttm_reserved_days", 'ttm_available_days', 'ttm_total_days', 'l90d_revenue']

        self.df = self.df.drop(columns=cols_to_drop, errors='ignore')
        return self

    def clean_columns(self):
        self.df['bedrooms'] = self.df['bedrooms'].fillna(0).astype(int)
        self.df['baths'] = self.df['baths'].fillna(0)
        return self

    def get_final_df(self):
        return self.df

    def train_test(self, target_col, test_size = .2, random_state = 0):
        self.target_col = target_col

        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test
