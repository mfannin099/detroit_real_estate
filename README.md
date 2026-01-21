# Detroit Airbnb Listings Dashboard

An interactive data visualization dashboard for exploring Airbnb listings in the Detroit metropolitan area. Built with Python Dash and Plotly, this application allows users to filter, analyze, and visualize short-term rental market data with real-time metrics and correlation analysis.

## Features

### 🔍 Interactive Filtering

* **Listing Type**: Filter by property type (apartment, house, villa, etc.)
* **Superhost Status**: View all listings, superhosts only, or non-superhosts
* **Bedrooms**: Range slider to filter by number of bedrooms (0-7+)
* **Bathrooms**: Range slider to filter by number of bathrooms (0-4.5+)
* **Reset Filters**: One-click button to reset all filters to default values

### 📊 Key Performance Indicators (KPIs)

Real-time metrics that update based on filtered data:
* **Average Overall Rating**: Mean rating across all filtered listings
* **Average Bedrooms**: Average number of bedrooms
* **Average Bathrooms**: Average number of bathrooms
* **Superhost Percentage**: Percentage of superhosts in filtered results
* **Trailing 12 Mo Revenue**: Average annual revenue across filtered listings
* **Reserved Days**: Average number of reserved days

### 📈 Data Visualizations

* **Sortable Data Table**: Browse listings with multi-column sorting capability
* **Interactive Map**: Visualize listing locations on an OpenStreetMap interface with hover details
* **Correlation Matrix**: Heatmap showing relationships between ratings, property features, and revenue metrics
* **Live Statistics**: Real-time count of filtered results vs. total listings

### 🗺️ Map Features

* Geospatial plotting of all filtered Airbnb listings
* Hover tooltips showing:
  + Listing name
  + Property type
  + Number of bedrooms and bathrooms
  + Overall rating
* Interactive zoom and pan capabilities

### 🔗 Correlation Analysis

Interactive correlation matrix displaying relationships between:
* Property features (photos, bedrooms, bathrooms)
* Rating dimensions (overall, accuracy, check-in, cleanliness, communication, location, value)
* Revenue metrics (trailing 12-month revenue)

## Technology Stack

* **Python 3.x**
* **Dash**: Web application framework
* **Plotly**: Interactive visualizations and mapping
* **Pandas**: Data manipulation and analysis
* **Parquet**: Efficient data storage format

## Installation

1. Clone the repository:

```bash
git clone https://github.com/mfannin099/detroit_real_estate.git
cd detroit_real_estate
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
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

* Property details (bedrooms, bathrooms, guests, photos)
* Host information and superhost status
* Geographic coordinates
* Ratings across multiple dimensions (overall, accuracy, cleanliness, communication, location, value, check-in)
* Revenue metrics (trailing 12-month revenue, reserved days, occupancy)
* Amenities and policies

## Project Structure

```
detroit_real_estate/
├── app.py                          # Main Dash application
├── utils.py                        # Data cleaning utilities
├── assets/
│   └── style.css                   # Custom CSS styling
├── data/
│   └── airroi_listings.parquet     # Listing data
├── notebooks/                      # Jupyter notebooks for analysis
├── mlruns/                         # MLflow experiment tracking
├── mlflow_LinReg.py               # Machine learning models
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## Key Components

### Data Processing (utils.py)

The `DataCleaner` class handles data preprocessing:
* Loading parquet files
* Dropping unnecessary columns
* Handling missing values in bedroom/bathroom columns
* Method chaining for clean, readable data pipeline

### Callbacks

The application uses Dash callbacks for reactive updates:
* `filter_table()`: Applies user-selected filters to the dataset
* `update_row_count()`: Updates the listing count display
* `update_all_metrics()`: Calculates and displays KPIs based on filtered data
* `update_map()`: Refreshes map markers based on filtered data
* `update_correlation_matrix()`: Generates correlation heatmap from filtered data
* `reset_filters()`: Resets all filters to default values

### Styling

Custom CSS provides:
* Responsive layout with maximum width constraints
* Hover effects on metric cards
* Professional color scheme and typography
* Consistent spacing and padding throughout the interface

## Machine Learning

The project includes MLflow integration for tracking machine learning experiments (`mlflow_LinReg.py`), enabling:
* Revenue prediction modeling
* Experiment tracking and comparison
* Model versioning and deployment

## Recent Updates

* Added KPI dashboard with 6 key metrics
* Implemented correlation matrix for rating and revenue analysis
* Created `DataCleaner` utility class for modular data preprocessing
* Added reset filters button for improved UX
* Enhanced styling with hover effects and responsive design
* Improved map visualization with better padding and layout

## Future Enhancements

Planned features:
* Predictive revenue modeling interface
* Additional data visualizations
* Enhanced filtering options
* Time-series analysis of pricing trends

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available for educational and analytical purposes.

## Acknowledgments

* Data provided by AirROI
* Built with Dash by Plotly
* Map tiles by OpenStreetMap contributors