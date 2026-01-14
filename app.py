##TODO: Allow for selection of individual rows
##TODO: Map of lat, long of flitered df
##TODO: Correlation matrix... with the rating columns...?
##TODO: allow for df sorting
##TODO: do something along the lines of.... viewing n listings of len(original_df)

from dash import Dash, html, dcc, dash_table, callback, Output, Input
import plotly.express as px
import pandas as pd
 
# Reading in data
df = pd.read_parquet("data/airroi_listings.parquet")
# TODO: Better drop columns (drop more columns)
df = df.drop(columns=['cover_photo_url'])
# Fill NaN values with 0
df['bedrooms'] = df['bedrooms'].fillna(0)
df['baths'] = df['baths'].fillna(0)

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
        ], className='filter-item')
    ], className='filters-container'),

    # Element for the Datatable
    dash_table.DataTable(
    id='listings-table', 
        data=df.to_dict('records'),
        columns=[{'name': col, 'id': col} for col in df.columns],
        page_size=10,
        style_table={'overflowX': 'auto'}
    ),

    # Text to display number of rows filtered to
    html.Div(id='row-count-display')

])


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

@callback(
    Output('row-count-display', 'children'),
    Input('listings-table', 'data')
)
def update_row_count(filtered_data):
    num_filtered = len(filtered_data)
    total = len(df)
    return f"Viewing {num_filtered:,} of {total:,} listings"


if __name__ == '__main__': # This checks if you ran this file directly (like python app.py). If you imported this file into another file, this block won't run. (Per Claude)
    app.run_server(debug=True)