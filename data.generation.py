# Re-import necessary libraries
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Seed for reproducibility
np.random.seed(42)

# Load the demand forecasting dataset again (since environment was reset)
def generate_demand_forecasting_data(n_rows):
    start_date = datetime(2023, 1, 1)
    data = []

    for i in range(n_rows):
        date = start_date + timedelta(days=random.randint(0, 365))
        product_id = f"P{random.randint(1, 100):03d}"
        sales_volume = random.randint(50, 200)
        price = round(random.uniform(0.8, 1.2), 2)
        promotion = random.choice([0, 1])
        seasonality = random.choice([0, 1])
        historic_trend = random.randint(20, 100)
        supplier_lead_time = random.randint(5, 15)
        weather_index = random.choice(["Cold", "Hot", "Mild"])
        avg_rating = round(random.uniform(3.0, 5.0), 1)
        social_media_buzz = random.randint(50, 200)
        holiday_indicator = random.choice([0, 1])
        product_lifecycle = random.choice(["Growth", "Maturity", "Decline"])

        data.append([date, product_id, sales_volume, price, promotion, seasonality, 
                     historic_trend, supplier_lead_time, weather_index, avg_rating, 
                     social_media_buzz, holiday_indicator, product_lifecycle])

    columns = ["Date", "Product_ID", "Sales_Volume", "Price", "Promotion", "Seasonality", 
               "Historic_Trend", "Supplier_Lead_Time", "Weather_Index", "Avg_Rating", 
               "Social_Media_Buzz", "Holiday_Indicator", "Product_Lifecycle"]
    
    return pd.DataFrame(data, columns=columns)

# Generate the demand forecasting dataset
demand_forecasting_df = generate_demand_forecasting_data(1000)

# Define function to calculate reorder point based on demand forecast, lead time, and safety stock
def calculate_reorder_point(demand_forecast, lead_time, safety_stock=50):
    return int(demand_forecast * (lead_time / 7) + safety_stock)

# Redefine Inventory Management Dataset with correlated columns
def generate_correlated_inventory_management_data(forecasting_df, n_rows):
    data = []
    for i in range(n_rows):
        row = forecasting_df.iloc[i]
        product_id = row["Product_ID"]
        date = row["Date"]
        demand_forecast = row["Sales_Volume"]  # Aligning forecast with sales volume
        
        current_inventory = random.randint(100, 1000)  # Keep current inventory independent for real-time adjustments
        lead_time = row["Supplier_Lead_Time"]  # Aligning lead time with supplier data from forecasting dataset
        supplier_reliability = round(np.clip(1 - (lead_time - 7) * 0.05, 0.7, 0.99), 2)  # Higher lead time = lower reliability
        reorder_point = calculate_reorder_point(demand_forecast, lead_time)  # Calculate reorder point based on forecast & lead time
        order_frequency = "Weekly"  # Keeping this simple, but could be variable
        warehouse_capacity = random.randint(500, 2000)
        holding_cost = round(random.uniform(0.3, 1.0), 2)
        return_rate = round(random.uniform(0.01, 0.05), 2)

        data.append([date, product_id, current_inventory, reorder_point, lead_time, 
                     order_frequency, demand_forecast, warehouse_capacity, 
                     supplier_reliability, holding_cost, return_rate])

    columns = ["Date", "Product_ID", "Current_Inventory", "Reorder_Point", "Lead_Time", 
               "Order_Frequency", "Demand_Forecast", "Warehouse_Capacity", 
               "Supplier_Reliability", "Holding_Cost", "Return_Rate"]
    
    return pd.DataFrame(data, columns=columns)

# Generate the corrected inventory management dataset
correlated_inventory_management_df = generate_correlated_inventory_management_data(demand_forecasting_df, 1000)

# Display the correlated datasets to the user
import ace_tools as tools; tools.display_dataframe_to_user(name="Correlated Inventory Management Dataset", dataframe=correlated_inventory_management_df)

# Return the correlated inventory management dataframe
correlated_inventory_management_df.head()
