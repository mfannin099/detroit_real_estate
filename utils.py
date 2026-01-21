import pandas as pd 

class DataCleaner:
    def __init__(self, path):
        self.patt = path
        self.df = None
    
    def load_data(self):
        self.df = pd.read_parquet("data/airroi_listings.parquet")
        return self

    def drop_columns(self):
        cols_to_drop = ["cover_photo_url", "listing_id", "host_id", "host_name",
                        "cohost_ids", "chost_names", 'registration', "instant_book",
                        "ttm_revenue_native", "ttm_blocked_days",  'ttm_revpar', 'ttm_revpar_native', 'ttm_adjusted_revpar',
                        'l90d_revenue', 'l90d_revenue_native', 'l90d_avg_rate', 'l90d_avg_rate_native',
                        'l90d_occupancy', 'l90d_adjusted_occupancy', 'l90d_revpar',
                        'l90d_revpar_native', 'l90d_adjusted_revpar',
                        'l90d_adjusted_revpar_native', 'l90d_reserved_days',
                        'l90d_blocked_days', 'l90d_available_days', 'l90d_total_days' ]

        self.df = self.df.drop(columns=cols_to_drop, errors='ignore')
        return self

    def clean_columns(self):
        self.df['bedrooms'] = self.df['bedrooms'].fillna(0).astype(int)
        self.df['baths'] = self.df['baths'].fillna(0)
        return self

    def get_final_df(self):
        return self.df
