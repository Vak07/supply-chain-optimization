import pandas as pd

def load_demand_data(filepath):
    return pd.read_csv(filepath)

def load_inventory_data(filepath):
    return pd.read_csv(filepath)

# Example usage
if __name__ == "__main__":
    demand_data = load_demand_data('data/demand_data.csv')
    inventory_data = load_inventory_data('data/inventory_data.csv')
