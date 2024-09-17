import pytest
from src.data_ingestion import load_demand_data

def test_load_demand_data():
    data = load_demand_data('data/demand_data.csv')
    assert data is not None, "Data should not be None"
    assert len(data) > 0, "Data should not be empty"
