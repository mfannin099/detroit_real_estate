import numpy as np
import sys
import os
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
        logging.FileHandler('training.log'),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)


# Begin ML Workflow

###################################
################################### BEGIN Data Cleaning
###################################

X_train, X_test, y_train, y_test = (DataCleaner(data_path)
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

###################################
################################### END Data Cleaning
###################################


## TODO Linear Regression

###################################
################################### END Linear Regression
###################################

model = LinearRegressionModel()
model.fit(X_train, y_train)
predictions = model.predict(X=X_test)
model.evaluation(X=X_test,y=y_test,predictions=predictions)

# Example calling the dictionary
logger.info(f"Model R² Score: {model.metrics['r2_score']:.2f}")

###################################
################################### END Linear Regression
###################################

## TODO Random Forest
## TODO XGBoost
