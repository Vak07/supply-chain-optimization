from src.models.lstm_demand_forecasting import build_lstm_model, train_lstm_model
from src.data_preprocessing.demand_preprocessing import preprocess_demand_data
from src.data_ingestion import load_demand_data

def run_training_pipeline():
    # Load and preprocess data
    demand_data = load_demand_data('data/demand_data.csv')
    X_train, y_train, scaler = preprocess_demand_data(demand_data)
    
    # Build and train model
    lstm_model = build_lstm_model(input_shape=(X_train.shape[1], 1))
    history = train_lstm_model(lstm_model, X_train, y_train)
    
    # Save model (add saving logic if necessary)
