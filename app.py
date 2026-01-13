from dash import Dash, html, dcc, dash_table, callback, Output, Input
import plotly.express as px
import pandas as pd
 
# Reading in data
df = pd.read_parquet("data/airroi_listings.parquet")
df = df.drop(columns=['cover_photo_url'])

app = Dash(__name__) # creates your Dash application object. Think of it as turning on your dashboard. (Per Claude)

# Define the layout
app.layout = html.Div([
    html.H1("Detroit Airbnb Listings Dash Application"),

    # Filters section
    html.Div([
        html.Div([
            html.Label("Room Type:"),
            dcc.Dropdown(
                id='room-type-filter',
                options=[{'label': rt, 'value': rt} for rt in df['room_type'].unique()],
                value=None,
                placeholder="All Room Types"
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

    html.H2([
        html.A(
            "Dataset Schema & Download →",
            href="https://www.airroi.com/data-portal/markets/detroit-united-states",
            target="_blank",
            className="data-link"
        )
    ]),

    dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{'name': col, 'id': col} for col in df.columns],
        page_size=10,
        style_table={'overflowX': 'auto'}
    )

])


@callback(
    Output('listings-table', 'data'),
    Input('room-type-filter', 'value'),
    Input('superhost-filter', 'value'),
    Input('bedrooms-filter', 'value'),
    Input('baths-filter', 'value')
)
def filter_table(room_type, superhost, bedrooms_range, baths_range):
    filtered_df = df.copy()
    
    if room_type:
        filtered_df = filtered_df[filtered_df['room_type'] == room_type]
    
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


if __name__ == '__main__': # This checks if you ran this file directly (like python app.py). If you imported this file into another file, this block won't run. (Per Claude)
    app.run_server(debug=True)