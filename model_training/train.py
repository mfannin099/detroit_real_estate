import numpy as np
import sys
import os
from sklearn.preprocessing import LabelEncoder
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from utils import DataCleaner, LinearRegressionModel
data_path = os.path.join(parent_dir, "data", "airroi_listings.parquet")

import logging
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        #logging.FileHandler('training.log'),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)


# Begin ML Workflow

###################################
################################### BEGIN Data Cleaning
###################################
ml_preprocessing = DataCleaner(data_path)

X_train, X_test, y_train, y_test = (ml_preprocessing
    .load_data()
    .drop_columns()
    .drop_columns_ml()
    .clean_columns()
    .clean_columns_ml()
    .train_test_ml(target_col='ttm_revenue')
)

# Log dataset sizes
logger.info(f"Training set size: {len(X_train)}")
logger.info(f"Test set size: {len(X_test)}")
logger.info(f"y_train size: {len(y_train)}")
logger.info(f"y_test size: {len(y_test)}")

# Log missing values
logger.info("Checking for missing values in X_train:")
logger.info(f"\n{X_train.isnull().sum()}")

logger.info("Checking for missing values in X_test:")
logger.info(f"\n{X_test.isnull().sum()}")

logger.info(f"Missing values in y_train: {y_train.isnull().sum()}")
logger.info(f"Missing values in y_test: {y_test.isnull().sum()}")

# Grabbing the encoding for dash app
label_encoders = ml_preprocessing.encoders
logger.info(f"Encoders available: {list(label_encoders.keys())}")
for encoder_name, encoder in label_encoders.items():
    logger.info(f"  {encoder_name}: {list(encoder.classes_)}")


###################################
################################### END Data Cleaning
###################################


###################################
################################### START Linear Regression
###################################

model = LinearRegressionModel()
model.fit(X_train, y_train, features = ['rating_overall', 'bedrooms', 'baths', 'listing_type'])
predictions = model.predict(X=X_test)
model.evaluation(X=X_test,y=y_test,predictions=predictions)
model.save_model('../models/linear_regression.pkl', encoders=label_encoders)


# Base model with all features as an example
# model2 = LinearRegressionModel()
# model2.fit(X_train, y_train)
# predictions = model2.predict(X=X_test)
# model2.evaluation(X=X_test,y=y_test,predictions=predictions)
# model2.print_equation(precision=2)

# # Example calling the dictionary
# logger.info(f"Model R² Score: {model2.metrics['r2_score']:.2f}")

# e = model2.print_equation(precision=2)


###################################
################################### END Linear Regression
###################################

## TODO Random Forest
## TODO XGBoost
