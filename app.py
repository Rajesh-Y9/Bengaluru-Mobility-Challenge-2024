import os
import json
import sys
from tracking import process_videos
from counting import count_video
from prediction import predict
import torch
import glob

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def delete_pt_files(directory):
    pt_files = glob.glob(os.path.join(directory, '*.pt'))
    for file in pt_files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")

def app(input_json, output_json):
    # Ensure 'runs' directory exists
    runs_dir = 'runs'
    ensure_directory(runs_dir)

    # Step 1: Process videos and generate tracking data
    tracking_json = process_videos(input_json)
    
    # Step 2: Count vehicles
    cam_id = list(json.load(open(input_json)).keys())[0]
    count_video(cam_id, runs_dir)

    # Step 3 : Predict
    cam_conf_json = os.path.join('cam_configs', f'{cam_id}.json')
    time_interval_json = os.path.join('runs',f'{cam_id}_time_intervals.json')
    model_file = os.path.join('prediction_models',f'{cam_id}.pt')

    cam_conf_json = os.path.join('cam_configs', f'{cam_id}.json')
    with open(cam_conf_json, 'r') as f:
        cam_conf = json.load(f)
    
    n_nodes = len(cam_conf['node_mapping'])
    n_edges = len(cam_conf['edge_mapping'])
    
    node_mapping = cam_conf["node_mapping"]
    edge_mapping = cam_conf['edge_mapping']
    # Create edge_index
    edge_index = torch.tensor([[node_mapping[s] for s, _ in edge_mapping.values()],
                                [node_mapping[t] for _, t in edge_mapping.values()]], dtype=torch.long)
    
    predict(cam_id,cam_conf_json, time_interval_json ,model_file,n_nodes=n_nodes,n_edges=n_edges,edge_index=edge_index)

    # Delete .pt files from runs directory
    delete_pt_files(runs_dir)

    # Step 4: Combine cumulative and predicted counts
    cumulative_file = os.path.join(runs_dir, f'{cam_id}_cumulative_counts.json')
    predicted_file = os.path.join(runs_dir, 'predicted_cumulative_counts.json')

    with open(cumulative_file, 'r') as f:
        cumulative_data = json.load(f)
    with open(predicted_file, 'r') as f:
        predicted_data = json.load(f)

    # Combine the data
    combined_data = {
        cam_id: {
            "Cumulative Counts": cumulative_data[cam_id]["Cumulative Counts"],
            "Predicted Counts": predicted_data
        }
    }

    # Save the combined output
    with open(output_json, 'w') as f:
        json.dump(combined_data, f, indent=2)

    print(f"Output saved to {output_json}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python app.py input.json output.json")
        sys.exit(1)

    input_json = sys.argv[1]
    output_json = sys.argv[2]
    app(input_json, output_json)