import json
import torch
from st_gen import ST_GEN
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from dataloader import TrafficDataset


def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

def load_tracking_history(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)


def inference(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        pred = model(dataloader.dataset,device)
    
    # Denormalize the predictions
    std_dev = dataloader.dataset.std_dev.to(device)
    mean = dataloader.dataset.mean.to(device)
    pred = torch.clamp(pred * (std_dev + 1e-8) + mean, min=0)  # Ensuring non-negative values
    
    return pred

def generate_cumulative_counts(pred, config):
    cumulative_counts = {}
    for edge_idx, edge in enumerate(config['edge_mapping']):
        cumulative_counts[edge] = {}
        for v_idx, vehicle in enumerate(['Bicycle', 'Bus', 'Car', 'LCV', 'Three Wheeler', 'Truck', 'Two Wheeler']):
            count = max(0, int(pred[:, edge_idx, v_idx].sum().item()))  # Ensuring non-negative counts
            cumulative_counts[edge][vehicle] = count
    return cumulative_counts



def predict(cam_id , cam_config,time_interval_file,model_file , n_nodes , n_edges , edge_index ):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = load_config(cam_config)
    # Load your model
    model = ST_GEN(in_channels=7, out_channels=7, n_nodes=n_nodes,n_edges=n_edges,edge_index=edge_index).to(device)
    
    # Load the checkpoint
    checkpoint = torch.load(model_file)
    # Extract only the model's state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    

    data = TrafficDataset(cam_id,time_interval_file,cam_config,120,120)
    test_dataloader = DataLoader(data, batch_size=1, shuffle=False)
    print(test_dataloader.dataset.x.shape)
    
    

    
    # Run inference
    pred = inference(model, test_dataloader,device)
    
    # Generate cumulative counts
    cumulative_counts = generate_cumulative_counts(pred, config)
    
    # Save results to JSON
    with open('runs/predicted_cumulative_counts.json', 'w') as f:
        json.dump(cumulative_counts, f, indent=2)
    
    
