# Detroit Airbnb Listings Dashboard

An interactive data visualization dashboard for exploring Airbnb listings in the Detroit metropolitan area. Built with Python Dash and Plotly, this application allows users to filter, analyze, and visualize short-term rental market data.

## Features

### 🔍 Interactive Filtering
- **Listing Type**: Filter by property type (apartment, house, villa, etc.)
- **Superhost Status**: View all listings, superhosts only, or non-superhosts
- **Bedrooms**: Range slider to filter by number of bedrooms (0-7+)
- **Bathrooms**: Range slider to filter by number of bathrooms (0-4.5+)

### 📊 Data Visualization
- **Sortable Data Table**: Browse listings with multi-column sorting capability
- **Interactive Map**: Visualize listing locations on an OpenStreetMap interface with hover details
- **Live Statistics**: Real-time count of filtered results vs. total listings

### 🗺️ Map Features
- Geospatial plotting of all filtered Airbnb listings
- Hover tooltips showing:
  - Listing name
  - Property type
  - Number of bedrooms and bathrooms
  - Overall rating
- Interactive zoom and pan capabilities

## Technology Stack

- **Python 3.x**
- **Dash**: Web application framework
- **Plotly**: Interactive visualizations and mapping
- **Pandas**: Data manipulation and analysis
- **Parquet**: Efficient data storage format

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mfannin099/detroit_real_estate.git
cd detroit_real_estate
```

2. Install required dependencies:
```bash
pip install dash plotly pandas
```

3. Ensure the data file exists at `data/airroi_listings.parquet`

## Usage

Run the application:
```bash
python app.py
```

The dashboard will be available at `http://127.0.0.1:8050/` in your web browser.

## Data Source

Dataset provided by [AirROI](https://www.airroi.com/data-portal/markets/detroit-united-states), containing comprehensive Airbnb listing information for the Detroit market including:
- Property details (bedrooms, bathrooms, guests)
- Host information and superhost status
- Geographic coordinates
- Ratings and reviews
- Revenue metrics and occupancy data
- Amenities and policies

## Project Structure

```
detroit_real_estate/
├── app.py                          # Main Dash application
├── assets/
│   └── style.css                   # Custom CSS styling
├── data/
│   └── airroi_listings.parquet     # Listing data
└── README.md                       # Project documentation
```

## Key Components

### Callbacks
The application uses Dash callbacks for reactive updates:
- `filter_table()`: Applies user-selected filters to the dataset
- `update_row_count()`: Updates the listing count display
- `update_map()`: Refreshes map markers based on filtered data

### Data Processing
- Handles missing values in bedroom/bathroom columns
- Converts data between DataFrame and dictionary formats for Dash components
- Supports real-time filtering without page reloads

## Future Enhancements

Planned features (see TODOs in code):
- [ ] Row selection in data table
- [ ] Correlation matrix for rating columns
- [ ] Additional data visualizations
- [ ] Enhanced filtering options

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available for educational and analytical purposes.

## Acknowledgments

- Data provided by AirROI
- Built with Dash by Plotly
- Map tiles by OpenStreetMap contributo