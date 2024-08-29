'''
Functions necessary for training the model with 'train_model.py' or model demo with 'demo_model_usage.py'
- parse_bench_file : parser for the .bench files
- convert_to_dgl : use information from the parser to create dgl graph object
- parse_line : extract ground truth and features from ground truth + features file
- load_features_to_dgl : include the extracted features to dgl object
- GNNSage : GNN model using SAGEConv layers
- GNNConv : GNN model using GraphConv layers
- BenchGraphDataset : function which loads in the dataset and calls functions necessary to preprocess the data
- collate : a custome collect function for dataloading
'''

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from torch.utils.data import Dataset
from dgl.nn.pytorch import SAGEConv
from dgl.nn.pytorch import GraphConv

TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float32

################## functions for training model ##########################

# Parser for the .bench files
def parse_bench_file(file_path):
    graph = {}
    with open(file_path, 'r') as file:
        for line in file:

            # Parsing the line
            line = line.strip()

            # Check if its a input decleartion
            if line.startswith('INPUT'):
                node_id = line.split('(')[-1].split(')')[0]
                graph[node_id] = {'type': 'input'}

            # Check if its a ouput decleartion
            elif line.startswith('OUTPUT'):
                node_id = line.split('(')[-1].split(')')[0]
                graph[node_id] = {'type': 'output'}

            # If its a gate
            if '=' in line:
                parts = line.split('=')
                node_id = parts[0].strip()
                expression = parts[1].strip()
                if '(' not in expression or ')' not in expression:
                    continue
                gate_type, connections = expression.split('(', 1)
                connections = connections[:-1] 

                # Attach type of node, gate type, and its connections
                graph[node_id] = {
                    'type': 'gate',
                    'gate_type': gate_type.upper(),
                    'connections': [x.strip() for x in connections.split(',')]
                }
            
            # Skip any commented lines
            elif line.startswith('#'):
                continue
            else:
                continue
    return graph

# Use information from the parser to create dgl graph object
def convert_to_dgl(graph):

    # Initialize mappings and lists for edges
    node_mapping = {node_id: idx for idx, node_id in enumerate(graph.keys())}
    edges_src = []
    edges_dst = []
    
    # Populate edges based on connection information
    for node_id, info in graph.items():
        if 'connections' in info: 
            src_idx = node_mapping[node_id]
            for conn in info['connections']:
                if conn in node_mapping: 
                    dst_idx = node_mapping[conn]

                    edges_src.append(dst_idx)
                    edges_dst.append(src_idx)

    # Create the directed DGL graph object
    g = dgl.graph((torch.tensor(edges_src), torch.tensor(edges_dst)))

    return g, node_mapping

# Extract ground truth and features from ground truth + features file
def parse_line(line):
    features = {}

    # Define pattern necessary to correctly extract information
    initial_pattern = re.compile(r'(Gate|Visited|Leaks|Fanin|Fanout)\s*=\s*([^,]+?)(?=,|$)')
    for match in initial_pattern.finditer(line):
        key, value = match.groups()
        value = value.strip()
        if value.lower() in ['true', 'false']:
            value = value.lower() == 'true'
        else:
            try:
                value = float(value)
            except ValueError:
                pass
        # Attach ground truth
        features[key] = value
    
    # Attach "Level", "KIF", and "Adj. Tag" as features
    special_pattern = re.compile(r'Level\s*=\s*(\d+)\s*KIF\s*=\s*(\d+)\s*Adj.\s*Tag\s*=\s*(\d+)')
    special_match = special_pattern.search(line)
    if special_match:
        level, kif, adj_tag = special_match.groups()
        features['Level'] = float(level)
        features['KIF'] = float(kif)
        features['Adj. Tag'] = float(adj_tag)

    return features

def load_features_to_dgl(dgl_graph, feature_file_path, node_mapping, graph):
    
    # Number of features extracted from parse_line for each node
    num_original_features = 6  

    # Number of additional features extracted later
    num_new_features = 3 

    num_features = num_original_features + num_new_features
    num_nodes = dgl_graph.num_nodes()
    features_tensor = torch.zeros((num_nodes, num_features), dtype=torch.float32)
    labels_tensor = torch.zeros(num_nodes, dtype=torch.float32) 

    # Mapping from gate types to integers
    gate_type_mapping = {'AND': 0, 'OR': 1, 'INV': 2, 'NAND': 3, 'NOR': 4, 'XOR': 5, 'XNOR': 6}  

    with open(feature_file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            # Define necessary pattern needed to correctly extract node ID
            node_id_match = re.search(r'Node\s+([^:\s]+):', line)
            
            if node_id_match:
                node_id = node_id_match.group(1)
                if node_id not in node_mapping:
                    continue  

                dgl_idx = node_mapping[node_id]
                node_features = [0] * num_original_features

                # Use parse_line function to extract features
                parsed_features = parse_line(line)

                # Update node_features based on parsed_features
                for key, value in parsed_features.items():
                    if key == 'Gate':
                        node_features[0] = gate_type_mapping.get(value, -1)  
                    elif key in ['Fanin', 'Fanout', 'Level', 'KIF', 'Adj. Tag']:
                        feature_index = ['Fanin', 'Fanout', 'Level', 'KIF', 'Adj. Tag'].index(key) + 1
                        node_features[feature_index] = value
                
                # Update the features tensor for the node
                features_tensor[dgl_idx, :num_original_features] = torch.tensor(node_features, dtype=torch.float32)
                if 'Leaks' in parsed_features:

                    # If leaks set to '1' otherwise '0'
                    labels_tensor[dgl_idx] = 1 if parsed_features['Leaks'] else 0

    # Additional feature extraction
    for node_id, info in graph.items():
        if node_id in node_mapping:
            dgl_idx = node_mapping[node_id]

            # Flag 'keyinput' nodes as a feature
            keyinput_feature = 1 if "keyinput" in node_id.lower() else 0

            # Flag 'input' nodes as a feature
            input_feature = 1 if info.get('type') == 'input' else 0

            # Feature 3: Check if 'type' is 'output'
            output_feature = 1 if info.get('type') == 'output' else 0

            # Attach additional features to the features tensor
            features_tensor[dgl_idx, num_original_features:] = torch.tensor([keyinput_feature, input_feature, output_feature], dtype=torch.float32)

    # Attach features and labels to the graph
    dgl_graph.ndata['feat'] = features_tensor
    dgl_graph.ndata['leak'] = labels_tensor

    return dgl_graph

# GNN model using SAGEConv layers
class GNNSage(nn.Module):
    """
    Basic GraphSAGE-based GNN class object. 
    Constructs the model architecture upon initialization. 
    """

    def __init__(self, in_feats, hidden_size, dropout, agg_type='mean'):
        """
        Initialize the model object. Establishes model architecture and relevant hypers ('in_feats', 'hidden_size', 'dropout', 'agg_type')

        - in_feat : input feature size : int
        - hidden_size : hidden layer dimension : int
        - dourpout : droupout rate : float
        - aggregation_function : type of aggregation function : str
        """
        
        super(GNNSage, self).__init__()
        
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(SAGEConv(in_feats, hidden_size, agg_type, activation=F.relu))
        # Output layer
        self.layers.append(SAGEConv(hidden_size, 1, agg_type))
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features):
        """
        Define forward step of netowrk. 
        Passing inputs through convolution, apply relu and dropout, then pass through second convolution.

        - param features : Input node representations : torch.tensor
        - returns Final layer representation in logits : torch.tensor
        """
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)

        return h

class GNNConv(nn.Module):
    """
    Basic GraphConv-based GNN class object. 
    Constructs the model architecture upon initialization. 
    """
    
    def __init__(self, in_feat, hidden_size, dropout):
        """
        Initialize the model object. Establishes model architecture and relevant hypers ('in_feats', 'hidden_size', 'dropout')

        - in_feat : input feature size : int
        - hidden_size : hidden layer dimension : int
        - dourpout : droupout rate : float
        """
        
        super(GNNConv, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(GraphConv(in_feat, hidden_size, activation=F.relu, allow_zero_in_degree=True))
        # Output layer
        self.layers.append(GraphConv(hidden_size, 1, allow_zero_in_degree=True))
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features):
        """
        Define forward step of netowrk. 
        Passing inputs through convolution, apply relu and dropout, then pass through second convolution.

        - param features : Input node representations : torch.tensor
        - returns Final layer representation in logits : torch.tensor
        """
            
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)

        return h
    
# Function which loads in the dataset and calls functions necessary to preprocess the data   
class BenchGraphDataset(Dataset):
    def __init__(self, bench_dir, feature_dir, max_files=None):
        self.graphs = []
        self.labels = []  
        self.node_maps = [] 
        
        bench_files = [f for f in os.listdir(bench_dir) if f.endswith('.bench')]
        if max_files:
            bench_files = bench_files[:max_files]

        # Load in the files containing ground truth and features and match to the .bench circuit files
        for bench_file in bench_files:
            file_name = bench_file.split('.')[0]
            feature_file_name = f"{file_name}_gt.txt"
            feature_file_path = os.path.join(feature_dir, feature_file_name)
            
            if not os.path.exists(feature_file_path):
                continue
            
            bench_file_path = os.path.join(bench_dir, bench_file)

            # Use .bench circuit file to create graph object
            graph = parse_bench_file(bench_file_path)            
            dgl_graph, node_map = convert_to_dgl(graph)
            
            # Load features to the dgl graph
            dgl_graph = load_features_to_dgl(dgl_graph, feature_file_path, node_map, graph)
            
            # Extract labels ("Leaks") to use as ground truth
            labels = self.extract_labels(dgl_graph, node_map)

            self.labels.append(labels)  
            self.graphs.append(dgl_graph)
            self.node_maps.append(node_map)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def extract_labels(self, dgl_graph, node_map):

        # Extract 'Leaks' as a label tensor
        if 'leak' in dgl_graph.ndata:
            labels = dgl_graph.ndata.pop('leak')
            return labels
        else:
            # Return a tensor of zeros if 'Leaks' information doesnt exist
            return torch.zeros((len(node_map),), dtype=torch.int64)
        
# Function that adds noise to node features
class AddNoiseToFeatures(object):
    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level

    def __call__(self, graph):
        noise = torch.randn_like(graph.ndata['feat']) * self.noise_level
        graph.ndata['feat'] += noise
        return graph

# A custome collect function for dataloading
def collate(samples):
    graphs, labels = map(list, zip(*samples))
    
    # Batch graphs
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.cat(labels, dim=0)
    
    return batched_graph, batched_labels

################## modified functions purely for demo output purpose ##########################

# Function which loads in the dataset and calls functions necessary to preprocess the data   
class BenchGraphDataset_demo(Dataset):
    def __init__(self, bench_dir, feature_dir, max_files=None):
        self.graphs = []
        self.labels = []  
        self.node_maps = [] 
        self.file_names = []
        
        bench_files = [f for f in os.listdir(bench_dir) if f.endswith('.bench')]
        if max_files:
            bench_files = bench_files[:max_files]

        # Load in the files containing ground truth and features and match to the .bench circuit files
        for bench_file in bench_files:
            file_name = bench_file.split('.')[0]
            feature_file_name = f"{file_name}_gt.txt"
            feature_file_path = os.path.join(feature_dir, feature_file_name)
            
            if not os.path.exists(feature_file_path):
                continue
            
            bench_file_path = os.path.join(bench_dir, bench_file)

            # Use .bench circuit file to create graph object
            graph = parse_bench_file(bench_file_path)            
            dgl_graph, node_map = convert_to_dgl(graph)
            
            # Load features to the dgl graph
            dgl_graph = load_features_to_dgl(dgl_graph, feature_file_path, node_map, graph)
            
            # Extract labels ("Leaks") to use as ground truth
            labels = self.extract_labels(dgl_graph, node_map)

            self.labels.append(labels)  
            self.graphs.append(dgl_graph)
            self.node_maps.append(node_map)
            self.file_names.append(file_name)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx], self.node_maps[idx], self.file_names[idx]

    def extract_labels(self, dgl_graph, node_map):

        # Extract 'Leaks' as a label tensor
        if 'leak' in dgl_graph.ndata:
            labels = dgl_graph.ndata.pop('leak')
            return labels
        else:
            # Return a tensor of zeros if 'Leaks' information doesnt exist
            return torch.zeros((len(node_map),), dtype=torch.int64)

# A custome collect function for dataloading
def collate_demo(samples):
    graphs, labels, node_map, file_names = map(list, zip(*samples))
    
    # Batch graphs
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.cat(labels, dim=0)
    
    return batched_graph, batched_labels, node_map, file_names

