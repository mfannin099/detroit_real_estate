from dash import Dash, html, dcc, dash_table, callback, Output, Input
import plotly.express as px
import pandas as pd
import os
import sys
from utils import DataCleaner, LinearRegressionModel
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

data_path = os.path.join(project_root, "data", "airroi_listings.parquet")
model_path = os.path.join(project_root, "models", "linear_regression.pkl")
 
# Reading in data
df = (DataCleaner(data_path)
      .load_data()
      .drop_columns()
      .clean_columns()
      .get_final_df())

# Loading in Pickle File with Linear Regression Model
linearRegression = LinearRegressionModel()
linearRegression.load_model(filepath ='models/linear_regression.pkl')

app = Dash(__name__) # creates your Dash application object. Think of it as turning on your dashboard. (Per Claude)

# Define the layout
app.layout = html.Div([

    # Header 1
    html.H1("Detroit Airbnb Listings Dash Application"),

    # Link to Schema and data
    html.H2([
        html.A(
        "Dataset Schema & Download →",
        href="https://www.airroi.com/data-portal/markets/detroit-united-states",
        target="_blank",
        className="data-link"
    )
    ]),

    # Filters section
    html.Div([
        html.Div([
            html.Label("Listing Type:"),
            dcc.Dropdown(
                id='listing-type-filter',
                options=[{'label': lt, 'value': lt} for lt in df['listing_type'].unique()],
                value=None,
                placeholder="All Listing Types"
            )
        ], className='filter-item'),
        
        html.Div([
            html.Label("Superhost:"),
            dcc.Dropdown(
                id='superhost-filter',
                options=[
                    {'label': 'All', 'value': 'all'},
                    {'label': 'Superhost Only', 'value': True},
                    {'label': 'Non-Superhost', 'value': False}
                ],
                value='all',
                placeholder="All"
            )
        ], className='filter-item'),
        
        html.Div([
            html.Label("Bedrooms:"),
            dcc.RangeSlider(
                id='bedrooms-filter',
                min=df['bedrooms'].min(),
                max=df['bedrooms'].max(),
                step=1,
                value=[df['bedrooms'].min(), df['bedrooms'].max()],
                marks={i: str(i) for i in range(int(df['bedrooms'].min()), int(df['bedrooms'].max())+1)}
            )
        ], className='filter-item'),
        
        html.Div([
            html.Label("Bathrooms:"),
            dcc.RangeSlider(
                id='baths-filter',
                min=df['baths'].min(),
                max=df['baths'].max(),
                step=0.5,
                value=[df['baths'].min(), df['baths'].max()],
                marks={i: str(i) for i in range(int(df['baths'].min()), int(df['baths'].max())+1)}
            )
        ], className='filter-item'),

        html.Div([
        html.Button('Reset Filters', id='reset-button', n_clicks=0)
    ], className='filter-item reset-button-container')


    ], className='filters-container'),


    #KPI's
    html.Div([
        html.Div([
            html.H4("Average Overall Rating"),
            html.Div(id='avg-rating-display', className='metric-value')
        ], className='metric-card'),

        html.Div([
        html.H4("Average Trailing 12 Mo Revenue"),
        html.Div(id='ttm-revenue-display', className='metric-value')
    ], className='metric-card'),

    html.Div([
        html.H4("Average Trailing 12 Mo Reserved Days"),
        html.Div(id='ttm-reserved-days-display', className='metric-value')
    ], className='metric-card'),

    ], className='metrics-container'),

    # Element for the Datatable
    dash_table.DataTable(
    id='listings-table', 
        data=df.to_dict('records'),
        columns=[{'name': col, 'id': col} for col in df.columns],
        page_size=10,
        style_table={'overflowX': 'auto'},
        sort_action='native',
        sort_mode='multi'
    ),

    # Text to display number of rows filtered to
    html.Div(id='row-count-display'),

    html.Hr(),

    # Map container
    html.Div([
        html.H3("Listing Locations"),
        dcc.Graph(id='listings-map')
    ], className = 'map-container'),

    html.Hr(),

    # Correlation Matrix
    html.Div([
        html.H3("Rating & Revenue Correlation Matrix"),
        dcc.Graph(id='correlation-matrix')
    ], className='correlation-container'),

    html.Hr(),

    # Display/ Allow user to Predict Revenue
    html.Div([
        html.H3("Predict Your Airbnb Revenue for 12 Months"),
        html.P("Enter property details to estimate trailing 12-month revenue:", 
               className='prediction-subtitle'),
        
        html.Div([
            # Rating Overall
            html.Div([
                html.Label("Overall Rating:"),
                dcc.Slider(
                    id='predict-rating',
                    min=1,
                    max=5,
                    step=0.5,
                    value=4.5,
                    marks={i: str(i) for i in range(1, 6)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], className='prediction-input'),
            
            # Bedrooms
            html.Div([
                html.Label("Bedrooms:"),
                dcc.Input(
                    id='predict-bedrooms',
                    type='number',
                    value=2,
                    min=0,
                    max=10,
                    className='prediction-number-input'
                ),
            ], className='prediction-input'),
        ], className='prediction-row'),
        
        html.Div([
            # Bathrooms
            html.Div([
                html.Label("Bathrooms:"),
                dcc.Input(
                    id='predict-baths',
                    type='number',
                    value=1,
                    min=0,
                    max=10,
                    step=0.5,
                    className='prediction-number-input'
                ),
            ], className='prediction-input'),
            
            # Listing Type
            html.Div([
                html.Label("Listing Type:"),
                dcc.Dropdown(
                    id='predict-listing-type',
                    options=[{'label': lt, 'value': lt} for lt in df['listing_type'].unique()],
                    value=df['listing_type'].iloc[0],
                    clearable=False,
                    className='prediction-dropdown'
                ),
            ], className='prediction-input'),
        ], className='prediction-row'),
        
        # Predict Button
        html.Div([
            html.Button(
                'Predict Revenue',
                id='predict-button',
                n_clicks=0,
                className='predict-button'
            ),
        ], className='predict-button-container'),
        
        # Prediction Output
        html.Div(id='prediction-output', className='prediction-output'),
        
        # Model Info
        html.Div([
            html.H4("📊 Model Information"),
            html.P(f"Features: {', '.join(linearRegression.feature_names)}"),
            html.P(f"R² Score: {linearRegression.metrics.get('r2_score', 0):.4f}" if linearRegression.metrics else ""),
            html.P(f"RMSE: ${linearRegression.metrics.get('rmse', 0):,.2f}" if linearRegression.metrics else "")
        ], className='model-info')
        
    ], className='prediction-container')

])



# Begin Callbacks
# Filters
@callback( # Decorator states that when one of the Inputs change, the function is ran
    Output('listings-table', 'data'),
    Input('listing-type-filter', 'value'),
    Input('superhost-filter', 'value'),
    Input('bedrooms-filter', 'value'),
    Input('baths-filter', 'value')
)
def filter_table(listing_type, superhost, bedrooms_range, baths_range):
    filtered_df = df.copy()
    
    if listing_type:
        filtered_df = filtered_df[filtered_df['listing_type'] == listing_type]

    # Default to 'all' if superhost is None
    if superhost is None:
        superhost = 'all'
    
    if superhost != 'all':
        filtered_df = filtered_df[filtered_df['superhost'] == superhost]
    
    filtered_df = filtered_df[
        (filtered_df['bedrooms'] >= bedrooms_range[0]) & 
        (filtered_df['bedrooms'] <= bedrooms_range[1])
    ]
    
    filtered_df = filtered_df[
        (filtered_df['baths'] >= baths_range[0]) & 
        (filtered_df['baths'] <= baths_range[1])
    ]
    
    return filtered_df.to_dict('records')

# Reset Filters
@callback(
    Output('listing-type-filter', 'value'),
    Output('superhost-filter', 'value'),
    Output('bedrooms-filter', 'value'),
    Output('baths-filter', 'value'),
    Input('reset-button', 'n_clicks')
)
def reset_filters(n_clicks):
    if n_clicks > 0:
        return None, 'all', [df['bedrooms'].min(), df['bedrooms'].max()], [df['baths'].min(), df['baths'].max()]
    return None, 'all', [df['bedrooms'].min(), df['bedrooms'].max()], [df['baths'].min(), df['baths'].max()]


# KPIS
@callback(
    Output('avg-rating-display', 'children'),
    Input('listings-table', 'data')
)
def update_avg_rating(filtered_data):
    filtered_df = pd.DataFrame(filtered_data)
    
    # Calculate average rating, handling potential NaN values
    if len(filtered_df) > 0 and 'rating_overall' in filtered_df.columns:
        avg_rating = filtered_df['rating_overall'].mean()
        if pd.notna(avg_rating):
            return f"{avg_rating:.2f}"
    
    return "N/A"

@callback(
    Output('ttm-revenue-display', 'children'),
    Input('listings-table', 'data')
)
def update_ttm_revenue(filtered_data):
    filtered_df = pd.DataFrame(filtered_data)
    
    if len(filtered_df) > 0 and 'ttm_revenue' in filtered_df.columns:
        avg_ttm_revenue = filtered_df['ttm_revenue'].mean()
        if pd.notna(avg_ttm_revenue):
            return f"${avg_ttm_revenue:,.0f}"
    
    return "N/A"

@callback(
    Output('ttm-reserved-days-display', 'children'),
    Input('listings-table', 'data')
)
def update_ttm_reserved_days(filtered_data):
    filtered_df = pd.DataFrame(filtered_data)
    
    if len(filtered_df) > 0 and 'ttm_reserved_days' in filtered_df.columns:
        avg_ttm_reserved_days = filtered_df['ttm_reserved_days'].mean()
        if pd.notna(avg_ttm_reserved_days):
            return f"{avg_ttm_reserved_days:,.0f}"
    
    return "N/A"

@callback(
    Output('row-count-display', 'children'),
    Input('listings-table', 'data')
)
def update_row_count(filtered_data):
    num_filtered = len(filtered_data)
    total = len(df)
    return f"Viewing {num_filtered:,} of {total:,} listings"


@callback(
    Output('listings-map', 'figure'),
    Input('listings-table', 'data')
)
def update_map(filtered_data):
    # Convert filtered data back to DataFrame
    filtered_df = pd.DataFrame(filtered_data)
    
    # Create the map
    fig = px.scatter_mapbox(
        filtered_df,
        lat='latitude',
        lon='longitude',
        hover_name='listing_name',
        hover_data={
            'listing_type': True,
            'bedrooms': True,
            'baths': True,
            'rating_overall': ':.2f',  # Format to 2 decimals
            'latitude': False,  # Hide lat/lon in hover
            'longitude': False
        },
        zoom=10,
        height=600,
    )
    
    # Use OpenStreetMap style (free, no token needed)
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )

    fig.update_traces(
        marker=dict(
            size=12,  # Fixed size for all markers
            opacity=0.8,  # Slight transparency
        )
    )
    
    return fig

# Correlation Matrix
@callback(
    Output('correlation-matrix', 'figure'),
    Input('listings-table', 'data')
)
def update_correlation_matrix(filtered_data):
    filtered_df = pd.DataFrame(filtered_data)
    
    # Select only the columns we want for correlation
    correlation_cols = [
        'photos_count', 'bedrooms', 'baths',
        'rating_overall', 'rating_accuracy', 'rating_checkin', 
        'rating_cleanliness', 'rating_communication', 'rating_location', 
        'rating_value', 'ttm_revenue'
    ]
    
    # Filter to only include columns that exist in the dataframe
    available_cols = [col for col in correlation_cols if col in filtered_df.columns]
    
    # Calculate correlation matrix
    corr_matrix = filtered_df[available_cols].corr()

    label_mapping = {
    'photos_count': 'Photos',
    'bedrooms': 'Bedrooms',
    'baths': 'Bathrooms',
    'rating_overall': 'Overall Rating',
    'rating_accuracy': 'Accuracy',
    'rating_checkin': 'Check-in',
    'rating_cleanliness': 'Cleanliness',
    'rating_communication': 'Communication',
    'rating_location': 'Location',
    'rating_value': 'Value',
    'ttm_revenue': 'Revenue (12mo)'
}

    corr_matrix = corr_matrix.rename(columns=label_mapping, index=label_mapping)
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale='RdBu_r',  # Red-Blue colorscale (reversed)
        zmin=-1,
        zmax=1,
        text_auto='.2f',  # Show correlation values with 2 decimals
        aspect='auto'
    )
    
    fig.update_layout(
        title="Correlation between Ratings and Revenue",
        height=600,
        margin={"r": 20, "t": 60, "l": 20, "b": 20}
    )
    
    return fig

# Prediction Callback
@callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    Input('predict-rating', 'value'),
    Input('predict-bedrooms', 'value'),
    Input('predict-baths', 'value'),
    Input('predict-listing-type', 'value')
)
def predict_revenue(n_clicks, rating, bedrooms, baths, listing_type):
    """Make revenue prediction using the trained model."""
    if n_clicks == 0:
        return html.Div([
            html.H3("Enter property details above and click Predict")
        ], className='prediction-placeholder')
    
    try:
        # Prepare input data
        input_dict = {
            'rating_overall': rating,
            'bedrooms': bedrooms,
            'baths': baths,
        }
        
        # Encode listing_type if encoder exists
        if hasattr(linearRegression, 'encoders') and 'listing_type' in linearRegression.encoders:
            try:
                encoded_listing = linearRegression.encoders['listing_type'].transform([listing_type])[0]
                input_dict['listing_type'] = encoded_listing
            except ValueError:
                input_dict['listing_type'] = 0
        
        # Create DataFrame
        input_data = pd.DataFrame([input_dict])
        
        # Add any missing features the model needs
        for feature in linearRegression.feature_names:
            if feature not in input_data.columns:
                input_data[feature] = 0
        
        # Ensure correct column order
        input_data = input_data[linearRegression.feature_names]
        
        # Make prediction
        prediction = linearRegression.predict(input_data)[0]
        
        # Display result
        return html.Div([
            html.H2(f"💰 Predicted Annual Revenue: ${prediction:,.2f}", 
                   className='prediction-result-amount'),
            html.Div([
                html.P(f" Rating: {rating}/5 | 🛏️ {bedrooms} bed | 🛁 {baths} bath | 📋 {listing_type}"),
            ], className='prediction-result-details')
        ], className='prediction-result-success')
    
    except Exception as e:
        return html.Div([
            html.H3(" Prediction Error", className='prediction-error-title'),
            html.P(str(e)),
            html.Pre(f"Expected features: {linearRegression.feature_names}", 
                    className='prediction-error-details')
        ], className='prediction-result-error')


if __name__ == '__main__': # This checks if you ran this file directly (like python app.py). If you imported this file into another file, this block won't run. (Per Claude)
    app.run_server(debug=True)