from src.data_ingestion import load_demand_data, load_inventory_data

def run_ingestion_pipeline():
    demand_data = load_demand_data('data/demand_data.csv')
    inventory_data = load_inventory_data('data/inventory_data.csv')
    return demand_data, inventory_data
