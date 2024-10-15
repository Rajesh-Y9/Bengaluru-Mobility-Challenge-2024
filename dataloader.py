import torch
import numpy as np
import pandas as pd
import os
from torch_geometric.data import InMemoryDataset, Data
from shutil import copyfile
import json




class TrafficDataset(InMemoryDataset):
    """
    Dataset for Graph Neural Networks.
    """
    def __init__(self, cam_id ,  json_file,config_file, n_hist, n_pred, transform=None, pre_transform=None):
        self.json_file = json_file
        self.n_hist = n_hist
        self.cam_id = cam_id
        self.n_pred = n_pred
        self.config_file = config_file
        self.n_node = 8
        self.processed_dir = os.path.join(os.getcwd(), 'runs')
        super().__init__(None, transform, pre_transform)
        self.data, self.slices , self.mean,self.std_dev= torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return [self.json_file]

    @property
    def processed_dir(self):
        return self._processed_dir
    
    @processed_dir.setter
    def processed_dir(self, value):
        self._processed_dir = value
        if not os.path.exists(self._processed_dir):
            os.makedirs(self._processed_dir)
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    

    def process(self):
        """
        Process the raw datasets into saved .pt dataset for later use.
        Note that any self.fields here wont exist if loading straight from the .pt file
        """
        node_features, edge_features, edge_index = self.load_and_process_data(self.cam_id,self.json_file,self.config_file)
        num_timesteps, num_nodes, num_vehicle_types = node_features.shape
        
        print(node_features.shape)
        print(edge_features.shape)
        n_window = self.n_hist + self.n_pred
          # Convert NumPy arrays to PyTorch tensors
        node_features = torch.FloatTensor(node_features)
        edge_features = torch.FloatTensor(edge_features)
        
            # Calculate mean and std for normalization
        self.node_mean = torch.mean(torch.FloatTensor(node_features), dim=(0, 1))
        self.node_std = torch.std(torch.FloatTensor(node_features), dim=(0, 1))
        self.edge_mean = torch.mean(torch.FloatTensor(edge_features), dim=(0, 1))
        self.edge_std = torch.std(torch.FloatTensor(edge_features), dim=(0, 1))
        
        # Normalize features
        node_features = (node_features - self.node_mean) / (self.node_std + 1e-8)
        edge_features = (edge_features - self.edge_mean) / (self.edge_std + 1e-8)
        
        mean = self.edge_mean
        std_dev = self.edge_std
        
        print(num_timesteps)
        print(n_window)

        sequences = []
        
        g = Data()
        g.__num_nodes__ = num_nodes
        g.edge_index = edge_index
        g.x = node_features

        sequences.append(g)
        
        print(g.x.shape)

        data, slices = self.collate(sequences)
        torch.save((data, slices,mean , std_dev), self.processed_paths[0])
        
        
    def load_and_process_data(self,cam_id, json_file, config_file):
        with open(json_file, 'r') as f:
            data = json.load(f)

        with open(config_file, 'r') as f:
            config = json.load(f)
            

        time_intervals = data[cam_id]['Time Intervals']
#         print(f"Processing data for camera: {cam_id}")
#         print(time_intervals)
        
        time_steps = sorted([int(t) for t in time_intervals.keys()])
    
        # Find the min and max timestamps
        min_time = min(time_steps)
        max_time = max(time_steps)

        # Create a continuous range of timestamps
        continuous_time_steps = range(min_time, max_time + 1)
        num_timesteps = len(continuous_time_steps)
        
        print(f"Number of timesteps: {num_timesteps}")
        print(f"Time range: {min_time} to {max_time}")
        
        # Get node and edge mappings from config
        node_mapping = config['node_mapping']
        edge_mapping = config['edge_mapping']

        num_nodes = len(node_mapping)
        num_edges = len(edge_mapping)
        num_vehicle_types = 7  # Bicycle, Bus, Car, LCV, Three Wheeler, Truck, Two Wheeler
        
        
        # Create node features
        node_features = np.zeros((num_timesteps, num_nodes, num_vehicle_types))

        # Create edge features
        edge_features = np.zeros((num_timesteps, num_edges, num_vehicle_types))
        print(edge_features.shape)
        
        vehicle_mapping = {
            'Bicycle': 0, 'Bus': 1, 'Car': 2, 'LCV': 3,
            'Three Wheeler': 4, 'Truck': 5, 'Two Wheeler': 6
        }

        # Create a reverse mapping for edges
        edge_index_mapping = {edge: idx for idx, edge in enumerate(edge_mapping.keys())}
        print(edge_index_mapping)
        
        for t in continuous_time_steps:
            t_index = t - min_time  # Adjust the index to start from 0
            if str(t) in time_intervals:
                interval_data = time_intervals[str(t)]
                for edge, counts in interval_data.items():
                    edge_idx = edge_index_mapping[edge]
                    for vehicle, count in counts.items():
                        v = vehicle_mapping[vehicle]
                        edge_features[t_index, edge_idx, v] = count
        
        for t in continuous_time_steps:
            t_index = t - min_time  # Adjust the index to start from 0
            if str(t) in time_intervals:
                interval_data = time_intervals[str(t)]
                for edge, counts in interval_data.items():
                    source, target = edge_mapping[edge]
                    source_idx, target_idx = node_mapping[source], node_mapping[target]
                    for vehicle, count in counts.items():
                        v = vehicle_mapping[vehicle]
                        node_features[t_index, target_idx, v] += count
                        node_features[t_index, source_idx, v] += count

        # Create edge_index
        edge_index = torch.tensor([[node_mapping[s] for s, _ in edge_mapping.values()],
                                   [node_mapping[t] for _, t in edge_mapping.values()]], dtype=torch.long)
        print(node_features.shape)
        print(edge_features.shape)
        return node_features, edge_features, edge_index
