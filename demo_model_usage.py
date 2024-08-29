'''
This program demonstrate potential usage using the best model saved
- Load in the .bench file and its features (assuming program which can extract these features exists so that process is not calculated here)
- Assest the circuit with the model saved
- Output network list with node_id, gate type, adj. tag, leakage and other relavent information at the end
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
from features_label import GNNSage, GNNConv, BenchGraphDataset_demo, collate_demo

TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float32

# Set path for the output log.txt file
current_path = os.getcwd()
file_path = os.path.join(current_path, 'out_log.txt')

# For demo puprose the demo bench file and its feature file are in the same directory
# Replace with your '/path_to_directory' if otherwise
current_path = os.getcwd()
demo_file = demo_feature = os.path.join(current_path, 'demo_bench')

demo_data = BenchGraphDataset_demo(demo_file, demo_feature)
demo_input = DataLoader(demo_data, batch_size=1, shuffle=False, collate_fn=collate_demo)

# Default parameter setting
in_feat = 9
hidden_size = 64
dropout = 0.3

# Initialize the model 
model = GNNSage(in_feat, hidden_size, dropout, agg_type='mean').to(TORCH_DEVICE)

# Load the saved model state
model.load_state_dict(torch.load(os.path.join(current_path, 'wBest15_model_auto2kT7.pth')))
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

        # Calculate accuracy for each class
        total_predictions_1 = (labels == 1).float().sum()
        total_predictions_0 = (labels == 0).float().sum()
        correct_predictions_1 = ((preds == 1) & (labels == 1)).float().sum()
        correct_predictions_0 = ((preds == 0) & (labels == 0)).float().sum() 
        accuracy_1 = correct_predictions_1 / total_predictions_1 if total_predictions_1 > 0 else 0
        accuracy_0 = correct_predictions_0 / total_predictions_0 if total_predictions_0 > 0 else 0

        total_predictions = labels.size(0)
        accuracy = correct_predictions / total_predictions

        node_map = node_map_list[0]
        inv_node_map = {value: key for key, value in node_map.items()}

        gate_type_mapping = {'AND': 0, 'OR': 1, 'INV': 2, 'NAND': 3, 'NOR': 4, 'XOR': 5, 'XNOR': 6}  
        inv_gate_type = {value: key for key, value in gate_type_mapping.items()}

        node_features = graph.ndata['feat'].tolist()
        total_KIP = sum(row[6] == 1 for row in node_features)

        print('\nCompleted key leakage assessment!')
        
        print(f"\n{file_name}")
        print('Network list:')

        total = 0
        num_gate = 0
        num_input = 0
        num_output = 0
        es_leak = 0
        ac_leak = 0
        neg = 0
        pos = 0

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

                if preds[node_id] == 0 and labels[node_id] == 1:
                    neg += 1
                elif preds[node_id] == 1 and labels[node_id] == 0:
                    pos += 1

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

        with open(file_path, 'a') as file:
            file.write(f'File name: {file_name}')
            file.write(f'\nTotal Keyinputs: {total_KIP:.4f}')
            file.write(f'\nModel assessment time: {end_time - start_time}')
            file.write(f'\nTotal nodes in circuit: {num_gate}')
            file.write(f'\nNode leaking key information (Estimated): {es_leak}')
            file.write(f'\nNode leaking key information (Actual): {ac_leak}')
            file.write(f'\nPos class accuracy: {accuracy_1:.4f}')
            file.write(f'\nNeg class accuracy: {accuracy_0:.4f}')
            file.write(f'\nOverall accuracy: {accuracy:.4f}')
            file.write(f'\nNumber of false negatives: {neg}\n\n\n')
