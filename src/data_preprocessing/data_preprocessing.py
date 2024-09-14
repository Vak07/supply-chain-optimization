import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

def preprocess_demand_data(df):
    """
    Preprocess the demand data by scaling features, encoding categorical variables,
    and handling missing values.
    
    Args:
    df (pd.DataFrame): Raw demand data.

    Returns:
    X (np.array): Features for training.
    y (np.array): Target variable (next step demand).
    scaler (MinMaxScaler): Scaler fitted on the features.
    """
    # Handle missing values (example: fill missing sales volumes with median)
    df['Sales_Volume'].fillna(df['Sales_Volume'].median(), inplace=True)
    
    # Encode categorical variables (e.g., Seasonality, Promotion)
    df['Seasonality'] = df['Seasonality'].fillna("Unknown")
    label_encoder = LabelEncoder()
    df['Seasonality'] = label_encoder.fit_transform(df['Seasonality'])
    
    # Feature scaling using MinMaxScaler
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[['Sales_Volume', 'Price', 'Promotion', 'Seasonality']])
    
    # Target: Next timestep sales volume (shift by 1)
    target = df['Sales_Volume'].shift(-1).dropna()
    scaled_features = scaled_features[:-1]  # Match lengths
    
    return np.array(scaled_features), np.array(target), scaler

def preprocess_inventory_data(df):
    """
    Preprocess the inventory data by scaling features and handling missing values.
    
    Args:
    df (pd.DataFrame): Raw inventory data.

    Returns:
    X (np.array): Scaled features for inventory.
    scaler (StandardScaler): Scaler fitted on the features.
    """
    # Handle missing values (example: fill missing inventory levels with median)
    df['Current_Inventory'].fillna(df['Current_Inventory'].median(), inplace=True)
    
    # Feature scaling using StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[['Current_Inventory', 'Reorder_Point', 'Lead_Time']])
    
    return scaled_data, scaler

def preprocess_lagged_features(df, lag=1):
    """
    Create lag features for time-series data, adding previous sales values as new features.
    
    Args:
    df (pd.DataFrame): Raw demand data.
    lag (int): Number of previous timesteps to include as features.

    Returns:
    pd.DataFrame: Dataframe with added lagged features.
    """
    for i in range(1, lag + 1):
        df[f'Sales_Lag_{i}'] = df['Sales_Volume'].shift(i)
    
    # Drop missing rows created due to lagging
    df.dropna(inplace=True)
    
    return df

# Example usage:
if __name__ == "__main__":
    # Load datasets
    demand_data = pd.read_csv('data/demand_data.csv')
    inventory_data = pd.read_csv('data/inventory_data.csv')
    
    # Preprocess demand data with lag features
    demand_data_lagged = preprocess_lagged_features(demand_data, lag=3)  # Example with 3 lags
    X_demand, y_demand, scaler_demand = preprocess_demand_data(demand_data_lagged)
    
    # Preprocess inventory data
    X_inventory, scaler_inventory = preprocess_inventory_data(inventory_data)
    
    # Example print statements
    print("Preprocessed Demand Data (first 5 rows):")
    print(X_demand[:5])
    print("\nPreprocessed Inventory Data (first 5 rows):")
    print(X_inventory[:5])
