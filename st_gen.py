import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GENConv, BatchNorm
from torch_geometric.data import Data, Batch
from ncps.torch import CfC

class ST_GEN(nn.Module):
    def __init__(self, in_channels, out_channels, n_nodes,n_edges,edge_index, dropout=0.3):
        super(ST_GEN, self).__init__()
        self.n_pred = 7
        self.edge_index= edge_index
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.dropout = dropout
        lstm1_hidden_size = 128
        lstm2_hidden_size = 256

        # Modify input channels to match your data
        self.gat1 = GENConv(in_channels=7, out_channels=64)
        self.bn1 = BatchNorm(64)
        self.gat2 = GENConv(in_channels=64, out_channels=128)
        self.bn2 = BatchNorm(128)
        self.gat3 = GENConv(in_channels=128, out_channels=128)
        self.bn3 = BatchNorm(128)
        
#         LSTM Layers
#         self.lstm1 = nn.LSTM(input_size=128 * n_nodes, hidden_size=lstm1_hidden_size, num_layers=2, batch_first=True, dropout=dropout)
#         self.lstm2 = nn.LSTM(input_size=lstm1_hidden_size, hidden_size=lstm2_hidden_size, num_layers=2, batch_first=True, dropout=dropout)
        
        # Define CfC layers
        self.cfc1 = CfC(
            input_size=128*n_nodes,
            units=lstm1_hidden_size,
            proj_size=lstm1_hidden_size,  # Project to the same size as hidden units
            return_sequences=True,
            batch_first=True,
            mixed_memory=False,  # Set as per your requirement
            mode="default",  # Set as per your requirement
            activation="lecun_tanh",  # Set as per your requirement
            
            backbone_dropout=dropout
        )

        self.cfc2 = CfC(
            input_size=lstm1_hidden_size,
            units=lstm2_hidden_size,
            proj_size=lstm2_hidden_size,  # Project to the same size as hidden units
            return_sequences=True,
            batch_first=True,
            mixed_memory=False,  # Set as per your requirement
            mode="default",  # Set as per your requirement
            activation="lecun_tanh",  # Set as per your requirement
            
            backbone_dropout=dropout
        )
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(lstm2_hidden_size, 512)
        self.fc2 = nn.Linear(512, self.n_edges * self.n_pred)
        

    def forward(self, data , device):
        x = data.x
        
        # apply dropout
        if device == 'cpu':
            x = torch.FloatTensor(x)
        else:
            x = torch.tensor(x, dtype=torch.float32, device=device)
        

        edge_index = self.edge_index
        edge_index = edge_index.to(device)
            
        batch_size = 1
        seq_length = x.size(0) // batch_size
        # Reshape x to [batch_size * seq_length, n_nodes, in_channels]
        x = x.view(batch_size * seq_length, self.n_nodes, -1)

        # Create a batch for GAT
        data_list = [Data(x=x_i, edge_index=edge_index) for x_i in x]
        batch = Batch.from_data_list(data_list)
        
        # Apply GAT layers
        x = F.relu(self.bn1(self.gat1(batch.x, batch.edge_index)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.bn2(self.gat2(x, batch.edge_index)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.bn3(self.gat3(x, batch.edge_index)))

        # Reshape back to [batch_size, seq_length, n_nodes * out_channels]
        x = x.view(batch_size, seq_length, -1)

        # Apply LSTMs
        x, _ = self.cfc1(x)
        x, _ = self.cfc2(x)

        # Apply fully connected layers with non-linearities
        x = F.relu(self.fc1(x.reshape(batch_size * seq_length, -1)))
        x = self.fc2(x)

        # Reshape to final output
        x = x.view(batch_size * seq_length, self.n_edges, self.n_pred)  # [1920, 12, 7]
        return x
