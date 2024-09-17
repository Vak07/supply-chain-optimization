# src/model_gnn.py

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd

class RouteOptimizationModel(nn.Module):
    def __init__(self, config):
        super(RouteOptimizationModel, self).__init__()
        self.config = config
        self.conv1 = GCNConv(self.config["gnn"]["input_dim"], self.config["gnn"]["hidden_dim"])
        self.conv2 = GCNConv(self.config["gnn"]["hidden_dim"], self.config["gnn"]["output_dim"])

    def forward(self, data):
        """Define the forward pass for the GNN."""
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def train_model(self, data):
        """Train the GNN model."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["gnn"]["learning_rate"])
        criterion = nn.MSELoss()

        for epoch in range(self.config["gnn"]["epochs"]):
            self.train()
            optimizer.zero_grad()
            output = self(data)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()

        print("GNN training completed.")

if __name__ == "__main__":
    config = ConfigParser()
    config.read("config/config.yaml")
    route_data = pd.read_csv(config["data"]["route_data_path"])

    # Assume route_data is already processed into a PyTorch Geometric Data object
    data = Data(x=torch.tensor(route_data.x, dtype=torch.float), edge_index=torch.tensor(route_data.edge_index, dtype=torch.long))
    gnn_model = RouteOptimizationModel(config)
    gnn_model.train_model(data)
