from dash import Dash, html, dcc, dash_table
import plotly.express as px
import pandas as pd
 
# Reading in data
df = pd.read_parquet("data/airroi_listings.parquet")
df = df.drop(columns=['cover_photo_url'])

app = Dash(__name__) # creates your Dash application object. Think of it as turning on your dashboard. (Per Claude)

# Define the layout
app.layout = html.Div([
    html.H1("Detroit Airbnb Listings Dash Application"),

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
        page_size=3,
        style_table={'overflowX': 'auto'}
    )

])



if __name__ == '__main__': # This checks if you ran this file directly (like python app.py). If you imported this file into another file, this block won't run. (Per Claude)
    app.run_server(debug=True)