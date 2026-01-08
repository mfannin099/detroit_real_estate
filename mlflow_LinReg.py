# ## To launch mlflow ui - mlflow ui


import pandas as pd
import numpy as np
import mlflow 
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Read data
df = pd.read_csv("data/detroit_open_data_portal_property_sales.csv")

## Removing Outliers in hopes to at least make this look semi-decent
mean = df['Sale Price'].mean()
std = df['Sale Price'].std()

std_threshold=2
lower_threshold = mean - (std_threshold * std)
upper_threshold = mean + (std_threshold * std)

outliers_mask = (df['Sale Price'] < lower_threshold) | (df['Sale Price'] > upper_threshold)
cleaned_df = df[~outliers_mask].copy()

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

        mlflow.set_tag("problem_type", "regression")
        mlflow.set_tag("model_type", "Linear_Regression")
        mlflow.set_tag("outlier_detection", "std_2")

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

        ## BEGINNING SUGGESTION TO SAVE PNG OF PLOTS (IK these are going to be horrible)
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual vs Predicted Sale Prices')
        plt.tight_layout()
        plt.savefig('actual_vs_predicted.png')
        mlflow.log_artifact('actual_vs_predicted.png')
        plt.close()

        # Log the model (IMPORTANT!) --> this acutally allows you to use the model later
        mlflow.sklearn.log_model(model, "linear_regression_model")

        return model

# End Helper functions
# ---------------------------------------------------


#####
##### Begin Main Script
#####

df_ = clean_data(cleaned_df)
print(df_.shape)
EXPERIMENT_NAME = "detroit_property_sales_3"

## Begin model experiments
# Initial Model
X = df_[['month']]
y = df_[['Sale Price']]

fit_model(X,y, EXPERIMENT_NAME, run_name='Baseline_month_only')

# Model 2
X = df_[['month', 'year']]
y = df_[['Sale Price']]

X = X.fillna(0)
fit_model(X,y, EXPERIMENT_NAME, run_name='month_and_year')

# Model 3
X = df_[['month', 'year', 'Latitude', 'Longitude']]
y = df_[['Sale Price']]

X = X.fillna(0)
fit_model(X,y, EXPERIMENT_NAME, run_name='month_and_year_lat_long')

# Model 4
X = df_[['month', 'year', 'Latitude', 'Longitude', "Zip Code"]]
y = df_[['Sale Price']]

X = X.fillna(0)
fit_model(X,y, EXPERIMENT_NAME, run_name='month_and_year_lat_long_zipcode')

# Model 5
X = df_[['month', 'year', 'Latitude', 'Longitude', 'Zip Code', 'Neighborhood_Encoded']]
y = df_[['Sale Price']]

X = X.fillna(0)
fit_model(X,y, EXPERIMENT_NAME, run_name='date_location_neighborhood')

# Model 6
X = df_[['month', 'year', 'Latitude', 'Longitude', 'Zip Code', 'Neighborhood_Encoded', 'Street_Name_Encoded']]
y = df_[['Sale Price']]

X = X.fillna(0)
fit_model(X,y, EXPERIMENT_NAME, run_name='date_location_neighborhood_street')




## using the model now to male a prediction

# run_id = "026c5368cfb149be9fc628529072a590"
# model = mlflow.sklearn.load_model(f"runs:/{run_id}/linear_regression_model")

# new_properties = pd.DataFrame({
#     'month': [6, 7, 8, 9, 10],
#     'year': [2026, 2026, 2026, 2026, 2026],
#     'Latitude': [42.3314, 42.3500, 42.3700, 42.3200, 42.3400],
#     'Longitude': [-83.0458, -83.0500, -83.0600, -83.0400, -83.0550],
#     'Zip Code': [48201, 48202, 48203, 48204, 48205],
#     'Neighborhood_Encoded': [10, 15, 20, 25, 30],
#     "Street_Name_Encoded": [35, 40, 45, 50, 55]
# }) # Per Claude

# predictions = model.predict(new_properties)
# print("Predicted Prices")
# for i, price in enumerate(predictions):
#     print(f"Property {i+1}: ${price[0]:,.2f}") 