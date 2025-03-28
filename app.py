import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import time
import random
from prophet import Prophet
from prophet.plot import plot_plotly

# Configuration
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Taipei Parking Dashboard", page_icon=':car:', layout="wide")

# Load Data
@st.cache_data
def load_data():
    data = pd.read_csv("taipei_parking_cleaned.csv")
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['occupancy_rate'] = data['total_vehicle'] / data['total_parking_space']
    return data

parking = load_data()

# Sidebar Filters
st.sidebar.header("Filters")
vehicle_type = st.sidebar.selectbox("Select Vehicle Type", ["All", "Car", "Scooter"])
year_range = st.sidebar.slider("Select Year Range", 
                              min_value=parking['year'].min(), 
                              max_value=parking['year'].max(), 
                              value=(parking['year'].min(), parking['year'].max()))

# Apply Filters
filtered_parking = parking[(parking['year'] >= year_range[0]) & (parking['year'] <= year_range[1])]

if vehicle_type == "Car":
    filtered_parking = filtered_parking.copy()
    filtered_parking['total_vehicle'] = filtered_parking['total_car']
    filtered_parking['income_count'] = filtered_parking['income_car_count']
elif vehicle_type == "Scooter":
    filtered_parking = filtered_parking.copy()
    filtered_parking['total_vehicle'] = filtered_parking['total_scooter']
    filtered_parking['income_count'] = filtered_parking['income_scooter_count']
else:
    filtered_parking = filtered_parking.copy()
    filtered_parking['total_vehicle'] = filtered_parking['total_car'] + filtered_parking['total_scooter']
    filtered_parking['income_count'] = filtered_parking['total_income_count']

# Dashboard Title
st.title("🚗 Taipei Parking Dashboard")

# Key Performance Indicators
st.header("Key Performance Indicators")
col_kpi1, col_kpi2, col_kpi3 = st.columns(3)

with col_kpi1:
    st.metric("Total Revenue (All Years)", f"${filtered_parking['income_count'].sum():,.2f}")

with col_kpi2:
    st.metric("Total Vehicles Parked", f"{filtered_parking['total_vehicle'].sum():,}")

with col_kpi3:
    st.metric("Busiest Year", filtered_parking.groupby('year')['occupancy_rate'].mean().idxmax())

# Parking Insights
st.header("📊 Parking Insights")
col1, col2 = st.columns(2)

with col1:
    title1 = f"{vehicle_type} Parking Availability Over Time"
    y_axis1 = (['car_parking_space', 'scooter_parking_space'] if vehicle_type == "All" else 
              ['car_parking_space'] if vehicle_type == "Car" else 
              ['scooter_parking_space'])
    fig1 = px.line(filtered_parking, x='date', y=y_axis1, title=title1)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    title2 = f"{vehicle_type} Growth Trends in Taipei"
    y_axis2 = (['total_car', 'total_scooter'] if vehicle_type == "All" else 
              ['total_car'] if vehicle_type == "Car" else 
              ['total_scooter'])
    fig2 = px.line(filtered_parking, x='date', y=y_axis2, title=title2)
    st.plotly_chart(fig2, use_container_width=True)

# Peak Parking Demand
monthly_demand = filtered_parking.groupby(['year', 'month'])['occupancy_rate'].mean().reset_index()
top_months = monthly_demand.groupby('year').apply(lambda x: x.nlargest(1, 'occupancy_rate')).reset_index(drop=True)

fig_peak_months = px.bar(monthly_demand, x='month', y='occupancy_rate', color='occupancy_rate',
                         animation_frame='year', 
                         category_orders={"month":["January", "February", "March", "April", "May", 
                                                 "June", "July", "August", "September", "October", 
                                                 "November", "December"]},
                         title="Peak Parking Demand by Month in Each Year",
                         labels={'occupancy_rate': 'Average Occupancy Rate', 'month': 'Month'})
st.plotly_chart(fig_peak_months, use_container_width=True)

st.write("### Top Busiest Parking Month Each Year")
st.dataframe(top_months[['year', 'month', 'occupancy_rate']])

# Revenue Insights
st.header("💰 Revenue Insights")
col3, col4 = st.columns(2)

with col3:
    fig_revenue = px.line(filtered_parking, x='date', y='income_count', 
                         title="💰 Monthly Parking Revenue Trends", markers=True)
    st.plotly_chart(fig_revenue, use_container_width=True)

with col4:
    yearly_revenue = filtered_parking.groupby('year')[['income_count']].sum().reset_index()
    fig_yearly_revenue = px.bar(yearly_revenue, x='year', y='income_count', 
                               title="💰 Yearly Parking Revenue Trends", text='income_count')
    st.plotly_chart(fig_yearly_revenue, use_container_width=True)

# Revenue Breakdown
st.header(":bar_chart: Revenue Breakdown")
revenue_breakdown = filtered_parking.groupby('year')['income_count'].sum().reset_index()
revenue_breakdown = revenue_breakdown.sort_values(by='year', ascending=True)
fig_revenue_breakdown = px.pie(revenue_breakdown, values='income_count', names='year', 
                              title="Revenue Contribution by Year")
st.plotly_chart(fig_revenue_breakdown, use_container_width=True)

# simulating data for 'real_time analysis.
def simulate_real_time():
    st.header("🚨 Real-Time Parking Monitor")
        
    live_spot = st.empty()
    chart_spot = st.empty()
    
    # Initialize data
    live_data = pd.DataFrame({
        'timestamp': [],
        'occupancy_rate': [],
        'vehicle_count': []
    })
    
    # Simulate 30 seconds of data
    for seconds in range(30):
        # Generate random data points
        new_row = pd.DataFrame({
            'timestamp': [pd.Timestamp.now()],
            'occupancy_rate': [random.uniform(0.5, 0.9)],
            'vehicle_count': [random.randint(100, 500)]
        })
        
        # Update the live data
        live_data = pd.concat([live_data, new_row]).tail(20)
        
        with live_spot.container():
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current Occupancy", f"{new_row['occupancy_rate'].iloc[0]*100:.1f}%", 
                         delta=f"{random.uniform(-2, 2):.1f}% from 1 min ago")
            with col2:
                st.metric("Vehicles in System", new_row['vehicle_count'].iloc[0], 
                         delta=random.randint(-10, 10))
        
        
        fig = px.line(live_data, x='timestamp', y='occupancy_rate', 
                     title="Live Occupancy Rate (Last 30 Seconds)")
        chart_spot.plotly_chart(fig, use_container_width=True)
        
        # Waiting time to 1 second
        time.sleep(1)

# button for triggering the simulation
if st.sidebar.button("Start Real-Time Simulation"):
    simulate_real_time()

st.sidebar.header("Advanced Forecast Options")
forecast_vehicle_type = st.sidebar.selectbox("Vehicle Type, parking space and revenue Predictions", 
                                          ["All", "Car", "Scooter", "Parking Spaces", "Revenue"])
forecast_view = st.sidebar.radio("View Forecast By", ["Monthly", "Yearly"])
forecast_years = st.sidebar.slider("Years to Forecast", 1, 10, 5)
include_confidence = st.sidebar.checkbox("Show Confidence Interval", True)

# Forecast Data
def prepare_forecast_data(forecast_type):
    if forecast_type == "Car":
        df = parking[['date', 'total_car', 'total_parking_space', 'income_car_count']].copy()
        df = df.rename(columns={'date': 'ds', 'total_car': 'y', 
                               'total_parking_space': 'capacity', 
                               'income_car_count': 'revenue'})
    elif forecast_type == "Scooter":
        df = parking[['date', 'total_scooter', 'total_parking_space', 'income_scooter_count']].copy()
        df = df.rename(columns={'date': 'ds', 'total_scooter': 'y',
                               'total_parking_space': 'capacity',
                               'income_scooter_count': 'revenue'})
    elif forecast_type == "Parking Spaces":
        df = parking[['date', 'total_parking_space']].copy()
        df = df.rename(columns={'date': 'ds', 'total_parking_space': 'y'})
    elif forecast_type == "Revenue":
        df = parking[['date', 'total_income_count']].copy()
        df = df.rename(columns={'date': 'ds', 'total_income_count': 'y'})
    else:  # "All"
        df = parking[['date', 'total_vehicle', 'total_parking_space', 'total_income_count']].copy()
        df = df.rename(columns={'date': 'ds', 'total_vehicle': 'y',
                               'total_parking_space': 'capacity',
                               'total_income_count': 'revenue'})
    return df

df_prophet = prepare_forecast_data(forecast_vehicle_type)

# Train Prophet Model with Additional Regressors
def train_prophet_model(df):
    model = Prophet()
    
    # Add additional regressors if available
    if 'capacity' in df.columns:
        model.add_regressor('capacity')
    if 'revenue' in df.columns:
        model.add_regressor('revenue')
    
    model.fit(df)
    return model

model = train_prophet_model(df_prophet)

# Create Future Dataframe with Regressors
future = model.make_future_dataframe(periods=forecast_years*365, freq='D')

# Add regressors to future dataframe
if 'capacity' in df_prophet.columns:
    # For simplicity, using last known capacity value for future predictions
    last_capacity = df_prophet['capacity'].iloc[-1]
    future['capacity'] = last_capacity
    
if 'revenue' in df_prophet.columns:
    # Using average of last 12 months revenue as future estimate
    avg_revenue = df_prophet['revenue'].tail(365).mean()
    future['revenue'] = avg_revenue

# Make Predictions
forecast = model.predict(future)

# Filter to only show future predictions
forecast_filtered = forecast[forecast['ds'] > df_prophet['ds'].max()]

# Aggregate Forecast Data
if forecast_view == "Yearly":
    forecast_filtered['Year'] = forecast_filtered['ds'].dt.year
    forecast_agg = forecast_filtered.groupby('Year')[['yhat', 'yhat_lower', 'yhat_upper']].mean().reset_index()
    x_col = 'Year'
    title = f"Yearly {forecast_vehicle_type} Forecast ({forecast_years} Years)"
else:
    forecast_filtered['Month'] = forecast_filtered['ds'].dt.to_period("M").astype(str)
    forecast_agg = forecast_filtered.groupby('Month')[['yhat', 'yhat_lower', 'yhat_upper']].mean().reset_index()
    x_col = 'Month'
    title = f"Monthly {forecast_vehicle_type} Forecast ({forecast_years} Years)"

# Display Forecast
st.header(f"🔮 {forecast_vehicle_type} Parking Forecast")
col_forecast1, col_forecast2 = st.columns([2, 1])

with col_forecast1:
    fig = px.line(forecast_agg, x=x_col, y='yhat', 
                 title=title,
                 labels={'yhat': 'Predicted Value', x_col: x_col})
    
    if include_confidence:
        fig.add_trace(go.Scatter(
            x=forecast_agg[x_col],
            y=forecast_agg['yhat_upper'],
            fill=None,
            mode='lines',
            line_color='rgba(255,0,0,0.1)',
            name='Upper Bound'
        ))
        fig.add_trace(go.Scatter(
            x=forecast_agg[x_col],
            y=forecast_agg['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(255,0,0,0.1)',
            name='Lower Bound'
        ))
    
    st.plotly_chart(fig, use_container_width=True)

with col_forecast2:
    st.write("### Forecast Summary")
    st.dataframe(forecast_agg.describe())
    
    # Calculate occupancy rate
    if forecast_vehicle_type in ["All", "Car", "Scooter"] and 'capacity' in df_prophet.columns:
        forecast_agg['occupancy_rate'] = forecast_agg['yhat'] / last_capacity
        st.write("### Predicted Occupancy Rates")
        st.dataframe(forecast_agg[[x_col, 'occupancy_rate']])


# Download Button
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(filtered_parking)
st.download_button(
    label="Download Taipei Parking Data as CSV",
    data=csv,
    file_name='Taipei_parking_data.csv',
    mime='text/csv',
)

# Download Forecast Data
forecast_csv = convert_df(forecast_filtered[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
st.download_button(
    label="Download Forecast Data as CSV",
    data=forecast_csv,
    file_name=f'taipei_parking_forecast_{forecast_vehicle_type}.csv',
    mime='text/csv',
)