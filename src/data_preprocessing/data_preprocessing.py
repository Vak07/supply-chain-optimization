# src/data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

class DataPreprocessing:
    def __init__(self, config):
        self.config = config

    def preprocess_demand_data(self, demand_data):
        """Preprocess the demand forecasting data."""
        # Handle missing values
        demand_data.fillna(method='ffill', inplace=True)

        # Feature scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        demand_data_scaled = scaler.fit_transform(demand_data)
        
        print("Demand forecasting data preprocessed.")
        return pd.DataFrame(demand_data_scaled), scaler

    def preprocess_inventory_data(self, inventory_data):
        """Preprocess the inventory management data."""
        # Handle missing values
        inventory_data.fillna(method='ffill', inplace=True)

        # Convert categorical features to numeric if necessary
        inventory_data = pd.get_dummies(inventory_data, drop_first=True)

        # Feature scaling
        scaler = StandardScaler()
        inventory_data_scaled = scaler.fit_transform(inventory_data)
        
        print("Inventory management data preprocessed.")
        return pd.DataFrame(inventory_data_scaled), scaler

    def preprocess_route_data(self, route_data):
        """Preprocess the route optimization data."""
        # Handle missing values
        route_data.fillna(0, inplace=True)

        # Normalize data for GNN
        scaler = MinMaxScaler(feature_range=(0, 1))
        route_data_scaled = scaler.fit_transform(route_data)

        print("Route optimization data preprocessed.")
        return pd.DataFrame(route_data_scaled), scaler

    def run(self, demand_data, inventory_data, route_data):
        """Run preprocessing for all datasets."""
        demand_data_processed, demand_scaler = self.preprocess_demand_data(demand_data)
        inventory_data_processed, inventory_scaler = self.preprocess_inventory_data(inventory_data)
        route_data_processed, route_scaler = self.preprocess_route_data(route_data)

        return demand_data_processed, inventory_data_processed, route_data_processed, demand_scaler, inventory_scaler, route_scaler

if __name__ == "__main__":
    from configparser import ConfigParser
    
    # Load configurations
    config = ConfigParser()
    config.read("config/config.yaml")

    # Load datasets
    demand_data = pd.read_csv(config["data"]["demand_data_path"])
    inventory_data = pd.read_csv(config["data"]["inventory_data_path"])
    route_data = pd.read_csv(config["data"]["route_data_path"])

    # Preprocess datasets
    preprocessor = DataPreprocessing(config)
    preprocessed_data = preprocessor.run(demand_data, inventory_data, route_data)
