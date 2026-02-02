import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler('training.log'),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, path):
        self.path = path
        self.df = None
        self.target_col = None
        self.encoders = {}
    
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

    def drop_columns_ml(self):
        cols_to_drop = ["guests", "listing_name", "latitude", "longitude", "extra_guest_fee",
        "amenities", 'cancellation_policy', 'currency', "ttm_avg_rate", 
        "ttm_reserved_days", 'ttm_available_days', 'ttm_total_days', 'l90d_revenue']

        self.df = self.df.drop(columns=cols_to_drop, errors='ignore')
        return self

    def clean_columns(self):
        self.df['bedrooms'] = self.df['bedrooms'].fillna(0).astype(int)
        self.df['baths'] = self.df['baths'].fillna(0)
        return self

    def clean_columns_ml(self):
        self.df = self.df.fillna(0)
        self.df['superhost'] = self.df['superhost'].map({True: 1, False: 0})
        return self

    def get_final_df(self):
        return self.df

    def train_test_ml(self, target_col, test_size = .2, random_state = 0):
        self.target_col = target_col

        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Label Encoding for listing_type
        le_listing = LabelEncoder()
        X_train['listing_type'] = le_listing.fit_transform(X_train['listing_type'])
        X_test['listing_type'] = le_listing.transform(X_test['listing_type'])
        self.encoders['listing_type'] = le_listing  # STORE 

        # Encode room_type
        le_room = LabelEncoder()
        X_train['room_type'] = le_room.fit_transform(X_train['room_type'])
        X_test['room_type'] = le_room.transform(X_test['room_type'])
        self.encoders['room_type'] = le_room  # STORE 
        
        return X_train, X_test, y_train, y_test



class LinearRegressionModel:

    def __init__(self):
        self.model = LinearRegression()
        self.feature_names = None 
        self.selected_features = None
        self.metrics = {}

    def fit(self, X_train, y_train, features=None):

        # Can call out what features are being used in the model class
        if features is not None:
            X_train_selected = X_train[features]
        else:
            X_train_selected = X_train

        
        # List of the features used
        self.feature_names = X_train_selected.columns.tolist()
        self.model.fit(X_train_selected, y_train)
        logger.info(f"Training complete with {len(self.feature_names)} features")

        return self
    
    def predict(self, X):
        try:
            X_selected = X[self.feature_names]
            predictions = self.model.predict(X_selected)
            logger.info(f"Predictions complete with {len(X)}")
            return predictions
        
        except Exception as e:
            raise ValueError(
                f"Prediction failed: {str(e)}. "
                f"Expected features: {self.feature_names}"
            )

    def evaluation(self, X,y,predictions):

        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)

        # Store metrics in the object
        self.metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mse': mse, 
            'n_samples': len(X)
        }

        logger.info("="*50)
        logger.info("MODEL EVALUATION METRICS")
        logger.info("="*50)
        logger.info(f"  RMSE (prediction error): ${rmse:,.2f}")
        logger.info(f"  MAE (avg error):         ${mae:,.2f}")
        logger.info(f"  R² (variance explained): {r2:.4f}")
        logger.info(f"  Samples evaluated:       {len(X)}")
        logger.info("="*50)

        return self.metrics
        
    def get_equation(self, precision=2):

        if self.feature_names is None:
            raise ValueError("Model must be fitted before getting equation")

        intercept = self.model.intercept_
        coefficients = self.model.coef_

        equation = f"y = {intercept:.{precision}f}"
    
        # Add each term
        for feature_name, coef in zip(self.feature_names, coefficients):
            # Handle positive/negative signs properly
            sign = "+" if coef >= 0 else "-"
            equation += f" {sign} {abs(coef):.{precision}f}*{feature_name}"
        
        return equation

    def print_equation(self, precision=2):

        equation = self.get_equation(precision)
        
        logger.info("="*50)
        logger.info("LINEAR REGRESSION EQUATION")
        logger.info("="*50)
        logger.info(equation)
        logger.info("-"*50)
        logger.info("Feature Coefficients:")
        
        # Show coefficients sorted by absolute value (importance)
        coef_dict = dict(zip(self.feature_names, self.model.coef_))
        sorted_coefs = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for feature, coef in sorted_coefs:
            logger.info(f"  {feature:20s}: {coef:+.{precision}f}")
        
        logger.info(f"  {'Intercept':20s}: {self.model.intercept_:+.{precision}f}")
        logger.info("="*50)

        return equation

    def save_model(self, filepath, encoders=None):

        if self.feature_names is None:
            raise ValueError("Model must be fitted before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'encoders': encoders if encoders is not None else {}
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
        if encoders:
            logger.info(f"Encoders saved: {list(encoders.keys())}")

    def load_model(self, filepath):
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.metrics = model_data.get('metrics', {})
        self.encoders = model_data.get('encoders', {})
        
        logger.info(f"Model loaded from {filepath}")
        logger.info(f"Features: {self.feature_names}")
        if self.encoders:
            logger.info(f"Encoders loaded: {list(self.encoders.keys())}")
        
        return self

