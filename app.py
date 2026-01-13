from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
 
# Reading in data
df = pd.read_parquet("data/airroi_listings.parquet")

app = Dash(__name__) # creates your Dash application object. Think of it as turning on your dashboard. (Per Claude)

# Define the layout
app.layout = html.Div([
    html.H1("My Airbnb Listings Dashboard")

])



if __name__ == '__main__': # This checks if you ran this file directly (like python app.py). If you imported this file into another file, this block won't run. (Per Claude)
    app.run_server(debug=True)