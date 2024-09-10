# src/model_lstm.py

import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class DemandForecastingModel:
    def __init__(self, config):
        self.config = config
        self.model = Sequential()

    def preprocess_data(self, data):
        """Preprocess the data for LSTM."""
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)
        X, y = [], []
        for i in range(self.config["lstm"]["look_back"], len(data_scaled)):
            X.append(data_scaled[i-self.config["lstm"]["look_back"]:i])
            y.append(data_scaled[i, 0])
        return np.array(X), np.array(y), scaler

    def build_model(self, input_shape):
        """Build LSTM model."""
        self.model.add(LSTM(50, input_shape=input_shape))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def train_model(self, X, y):
        """Train LSTM model."""
        self.build_model((X.shape[1], X.shape[2]))
        self.model.fit(X, y, epochs=self.config["lstm"]["epochs"], batch_size=self.config["lstm"]["batch_size"], verbose=2)

    def save_model(self):
        """Save the LSTM model."""
        self.model.save(self.config["lstm"]["model_path"])

    def run(self, data):
        """Run the demand forecasting model."""
        X, y, scaler = self.preprocess_data(data)
        self.train_model(X, y)
        self.save_model()
        return self.model, scaler

if __name__ == "__main__":
    config = ConfigParser()
    config.read("config/config.yaml")
    df_model = DemandForecastingModel(config)
    demand_data = pd.read_csv(config["data"]["demand_data_path"])
    df_model.run(demand_data)
