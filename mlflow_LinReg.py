## To launch mlflow ui - mlflow ui


import pandas as pd
import numpy as np
import mlflow 
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Read data
df = pd.read_csv("data/detroit_open_data_portal_property_sales.csv")

# Start Helper functions
# ---------------------------------------------------

def clean_data(df):
    df = df[df['Property Class Code'] == 201]
    df = df[df['Sale Price'] != 0]

    df["Sale Date"] = pd.to_datetime(df["Sale Date"])
    # Extract month and year
    df["month"] = df["Sale Date"].dt.month
    df["year"] = df["Sale Date"].dt.year

    # Column Cleaning when needed
    # Zip Code column remove whats after xxxxx-xxxx (remove this part: -xxxx)
    df['Zip Code'] = df['Zip Code'].astype(str).str.split('-').str[0]
    df['Zip Code'] = pd.to_numeric(df['Zip Code'], errors='coerce')

    # Labeling Encoding (I Know this is wrong but its easier to just grab this column)
    le = LabelEncoder()
    df['Neighborhood_Encoded'] = le.fit_transform(df['Neighborhood'])

    le2 = LabelEncoder()
    df['Grantor_Encoded'] = le2.fit_transform(df['Grantor'])

    le3 = LabelEncoder()
    df['Street_Name_Encoded'] = le3.fit_transform(df['Street Name'])

    return df

def fit_model(X,y, experiment_name, run_name=None):

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Mlflow logs
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples_train", X_train.shape[0])
        mlflow.log_param("n_samples_test", X_test.shape[0])
        mlflow.log_param("features", list(X.columns))

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # Log metrics
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)

        # Log the model (IMPORTANT!)
        mlflow.sklearn.log_model(model, "linear_regression_model")

        # # Inspect the learned formula: y = mx + b
        # print(f"Coefficient (slope): {model.coef_}")
        # print(f"Intercept: {model.intercept_}")

        return model

# End Helper functions
# ---------------------------------------------------


#####
##### Begin Main Script
#####

df = clean_data(df)
print(df.shape)
EXPERIMENT_NAME = "detroit_property_sales_"

## Begin model experiments
# Initial Model
X = df[['month']]
y = df[['Sale Price']]

fit_model(X,y, EXPERIMENT_NAME, run_name='Baseline_month_only')

# Model 2
X = df[['month', 'year']]
y = df[['Sale Price']]

X = X.fillna(0)
fit_model(X,y, EXPERIMENT_NAME, run_name='month_and_year')

# Model 3
X = df[['month', 'year', 'Latitude', 'Longitude']]
y = df[['Sale Price']]

X = X.fillna(0)
fit_model(X,y, EXPERIMENT_NAME, run_name='month_and_year_lat_long')

# Model 4
X = df[['month', 'year', 'Latitude', 'Longitude', "Zip Code"]]
y = df[['Sale Price']]

X = X.fillna(0)
fit_model(X,y, EXPERIMENT_NAME, run_name='month_and_year_lat_long_zipcode')

# Model 5
X = df[['month', 'year', 'Latitude', 'Longitude', 'Zip Code', 'Neighborhood_Encoded']]
y = df[['Sale Price']]

X = X.fillna(0)
fit_model(X,y, EXPERIMENT_NAME, run_name='date_location_neighborhood')

# Model 6
X = df[['month', 'year', 'Latitude', 'Longitude', 'Zip Code', 'Neighborhood_Encoded', 'Street_Name_Encoded']]
y = df[['Sale Price']]

X = X.fillna(0)
fit_model(X,y, EXPERIMENT_NAME, run_name='date_location_neighborhood_street')