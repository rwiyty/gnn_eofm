'''
This program demonstrate potential usage with CLI using the selected model
- From the CLI aquire folder name (with data) and model name
- Load in the .bench file and its features (assuming program which can extract these features exists so that process is not calculated here)
- Assest the circuit with the selected model
- Output network list with node_id, gate type, adj. tag, leakage and other relavent information to terminal
e.g., python GNN_EOFM.py --f demo_bench --m wBest15_model_auto2kT7.pth
'''

import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from features_label import GNNSage, GNNConv, BenchGraphDataset_demo, collate_demo

# Argument parsing
parser = argparse.ArgumentParser(description='Run key leakage assessment on a given circuit.')
parser.add_argument('--f', type=str, required=True, help="Folder name containing the dataset and features. Assumes the current path.")
parser.add_argument('--m', type=str, required=True, help='Name of the best model file (e.g., wBest15_model_auto2kT7.pth).')
args = parser.parse_args()

TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float32

# For demo puprose the demo bench file and its feature file are in the same directory
# Replace with your '/path_to_directory' if otherwise
current_path = os.getcwd()
demo_file = demo_feature = os.path.join(current_path, args.f)

demo_data = BenchGraphDataset_demo(demo_file, demo_feature)
demo_input = DataLoader(demo_data, batch_size=1, shuffle=False, collate_fn=collate_demo)

# Default parameter setting
in_feat = 9
hidden_size = 64
dropout = 0.3

# Initialize the model 
model = GNNSage(in_feat, hidden_size, dropout, agg_type='mean').to(TORCH_DEVICE)

# Load the saved model state
model.load_state_dict(torch.load(os.path.join(current_path, args.m)))
model.eval()

with torch.no_grad(): 
    for graph, labels, node_map_list, file_name in demo_input:

        graph = graph.to(TORCH_DEVICE)
        labels = labels.to(TORCH_DEVICE).float()

        print('Launching key leakage assessment...')

        # Start timing
        start_time = time.time()

        logits = model(graph, graph.ndata['feat']).squeeze()

        # Convert logits to probabilities with sigmoid
        preds = (torch.sigmoid(logits) >= 0.5).float()

        # End timing
        end_time = time.time()

        # Output calculation/preperation
        correct_predictions = (preds == labels).float().sum()
        total_predictions = labels.size(0)
        accuracy = correct_predictions / total_predictions

        node_map = node_map_list[0]
        inv_node_map = {value: key for key, value in node_map.items()}

        gate_type_mapping = {'AND': 0, 'OR': 1, 'INV': 2, 'NAND': 3, 'NOR': 4, 'XOR': 5, 'XNOR': 6}  
        inv_gate_type = {value: key for key, value in gate_type_mapping.items()}

        print('\nCompleted key leakage assessment!')

        print(f"\n{file_name}")
        print('Network list:')

        total = 0
        num_gate = 0
        num_input = 0
        num_output = 0
        es_leak = 0
        ac_leak = 0

        # Iterating over all nodes
        for node_id in range(graph.number_of_nodes()):
            
            total += 1
            node_features = graph.ndata['feat'][node_id].tolist()

            # Only output information on nodes that are gates
            if node_features[5] > 0:
                if preds[node_id] == 1:
                    leaka = 'true'
                    es_leak += 1
                else:
                    leaka = 'false'

                if labels[node_id] == 1:
                    leakb = 'true'
                    ac_leak += 1
                else:
                    leakb = 'false'

                print(f"Node {inv_node_map[node_id]}: Gate = {inv_gate_type[int(node_features[0])]}, Adj. tag = {int(node_features[5])}, Leaks (Estimated | Actual) = {leaka} | {leakb}")
                num_gate += 1
            elif node_features[6]:
                num_input += 1
        
        print(f'\nModel assessment time: {end_time - start_time}')
        print(f'PIs: {num_input}')
        print(f'POs: {total - num_input - num_gate}')
        print(f'Total nodes in circuit: {num_gate}')
        print(f'Node leaking key information (Estimated): {es_leak}')
        print(f'Node leaking key information (Actual): {ac_leak}')
        print(f'Overall accuracy: {accuracy:.4f}')
