# src/data_ingestion.py

import pandas as pd
from configparser import ConfigParser

class DataIngestion:
    def __init__(self, config):
        self.config = config

    def load_datasets(self):
        """Load datasets for demand forecasting, inventory management, and route optimization."""
        demand_data = pd.read_csv(self.config["data"]["demand_data_path"])
        inventory_data = pd.read_csv(self.config["data"]["inventory_data_path"])
        route_data = pd.read_csv(self.config["data"]["route_data_path"])
        print("Datasets loaded successfully.")
        return demand_data, inventory_data, route_data

    def run(self):
        """Run data ingestion pipeline."""
        demand_data, inventory_data, route_data = self.load_datasets()
        return demand_data, inventory_data, route_data

if __name__ == "__main__":
    config = ConfigParser()
    config.read("config/config.yaml")
    ingestion = DataIngestion(config)
    demand_data, inventory_data, route_data = ingestion.run()


