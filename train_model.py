'''
This code comebined with features_label.py trains the model to estimate key leakage in intergrated circuits under EOFM condition
- Set the necessary path and parameter
- Run the program and it will begin training the model
- The program will save the best model and plot the F1 score curve for training and validation at the end
'''

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.transforms as T
from torch.utils.data import DataLoader, Subset
from features_label import GNNSage, GNNConv, BenchGraphDataset, AddNoiseToFeatures, collate

TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float32
current_path = os.getcwd()

################## train/val/test data path ##########################
'''
Section where you can define the directory for dataset and ground truth + feature 
The test_dir and its feature_dir WONT be needed if not using a designated test set
See below in dataloading section for splitting test set instead
'''

# path to the dataset and ground truth + feature directory
bench_dir = os.path.join(current_path, 'Datasets/data_total_xor4k')
feature_dir = os.path.join(current_path, 'Datasets/data_total_xor4k/g2_gt')

# path to the dataset and ground truth + feature directory for test set
test_dir = os.path.join(current_path, 'Datasets/test_comprehensive')
test_feature_dir = os.path.join(current_path, 'Datasets/test_comprehensive/g2_gt')

# path to the best model
best_model_path = os.path.join(current_path, 'best_model.pth')

################## parameters ##########################
'''
Parameters setting section description:

- model_type : select the model using GraphCONV or SAGEConv ('conv', 'sage') : str
- in_feat : input feature size (shouldn't change unless modification to the feature extraction process was made) : int
- hidden_size : hidden layer dimension, adjust as needed : int
- dourpout : droupout rate (0-1) : float
- batch_size : the batch size for the model to train on, adjust as needed : int
- aggregation_function : aggregation function which ONLY APPLIES TO GraphSAGE model ('mean', 'pool', 'gcn', 'lstm') : str
- lr : learning rate, adjust as needed : float
- weight_decay_value : weight decay, adjust as needed : float
- num_epoch : number of training epoch, adjust as needed : int
- positive_weight : weight on the positive examples, adjust as needed : int/float
'''
model_type = 'sage'
in_feat = 9
hidden_size = 64
dropout = 0.3
batch_size = 16
aggregation_function = 'mean'
lr = 0.00005
weight_decay_value = 1e-6
num_epochs = 300
positive_weight = 15

################## start of main ##########################
'''
No additional modification should be needed beyond this point
Run the program and the model should begin to train
It will save the best model to the specified path and output the f1 score graph once completed
'''
# loading the model
if model_type == 'conv':
    print("Fetching model using GraphCONV...")
    model = GNNConv(in_feat, hidden_size, dropout).to(TORCH_DEVICE)  # Ensure model is on the right device
    print("Model successfully loaded!")
elif model_type == 'sage':
    print("Fetching model using GraphSAGE...")
    model = GNNSage(in_feat, hidden_size, dropout, agg_type='mean').to(TORCH_DEVICE)  # Ensure model is on the right device
    print("Model successfully loaded!")
else:
    print("invalid model type!")
    sys.exit()

################## data loading ##########################
print("\nLoading in dataset...")

# Initialize the dataset
dataset = BenchGraphDataset(bench_dir, feature_dir)

# Set the seed for shuffling the dataset for reproducibility
torch.manual_seed(42)
np.random.seed(42)

total_size = len(dataset)
train_ratio = 0.8
val_ratio = 1 - train_ratio

# If you are not using a designated test set *UNCOMMENT* the following code
# test_ratio = 1 - train_ratio - val_ratio

# Calculate the size of each split
train_size = int(train_ratio * total_size)
val_size = int(val_ratio * total_size)

# If you are not using a designated test set *UNCOMMENT* the following code
#test_size = total_size - train_size - val_size

# Generate a shuffled list of indices
indices = torch.randperm(total_size).tolist()

# Create subsets for training, validation, and testing
train_dataset = Subset(dataset, indices[:train_size])

# If you are not using a designated test set *COMMENT OUT* the following code
val_dataset = Subset(dataset, indices[train_size:])

# Introducing feature noise
transform = T.Compose([AddNoiseToFeatures(noise_level=0.1)])
transformed_data = [(transform(graph), label) for graph, label in train_dataset]

# If you are not using a designated test set *UNCOMMENT* the following code
#val_dataset = Subset(dataset, indices[train_size:train_size+val_size])
#test_dataset = Subset(dataset, indices[train_size+val_size:])

# Initialize DataLoaders for each split
train_dataloader = DataLoader(transformed_data, batch_size, shuffle=True, collate_fn=collate)
val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=collate)

# If you are not using a designated test set *COMMENT OUT* the following code
test_dataset = BenchGraphDataset(test_dir, test_feature_dir)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=collate)

# If you are not using a designated test set *UNCOMMENT* the following code
#test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate)

print("Dataset successfully loaded!")


################## start training ##########################
print("\nBegin training...\n")

optimizer = optim.Adam(model.parameters(), lr, weight_decay = weight_decay_value)

# Initialize the BCEWithLogitsLoss function
pos_weight = torch.tensor([positive_weight]).to(TORCH_DEVICE)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Initialize lists to store F1 scores for each epoch
train_f1_1_scores = []
val_f1_1_scores = []

# Initialize the best validation loss to infinity for best model logic
best_val_loss = float('inf')  

# Training loop
for epoch in range(num_epochs):
    model.train()  
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    correct_positives = 0  
    total_positives = 0 
    correct_negatives = 0  
    total_negatives = 0  

    for batched_graph, labels in train_dataloader:
        batched_graph = batched_graph.to(TORCH_DEVICE)
        labels = labels.to(TORCH_DEVICE).float() 

        optimizer.zero_grad()       
        logits = model(batched_graph, batched_graph.ndata['feat']).squeeze()
        loss = loss_fn(logits, labels)

        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        # Convert logits to probabilities with sigmoid
        probs = torch.sigmoid(logits)

        # Convert probabilities to binary predictions
        preds = (logits >= 0.5).float()

        # Calculate accuracy
        correct_predictions += (preds == labels).float().sum()
        total_predictions += labels.size(0)

        correct_positives += ((preds == 1) & (labels == 1)).float().sum().item()
        total_positives += (labels == 1).float().sum().item()
        correct_negatives += ((preds == 0) & (labels == 0)).float().sum().item()
        total_negatives += (labels == 0).float().sum().item()

    avg_loss = total_loss / len(train_dataloader)
    accuracy = correct_predictions / total_predictions

    # Calculate class-specific accuracies
    positive_accuracy = correct_positives / total_positives if total_positives > 0 else 0
    negative_accuracy = correct_negatives / total_negatives if total_negatives > 0 else 0

    print(f'Train - Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    print(f'Train - Correct 1\'s (Positives) Accuracy: {positive_accuracy:.4f}')
    print(f'Train - Correct 0\'s (Negatives) Accuracy: {negative_accuracy:.4f}')

    # Calculating Precision, Recall, and F1 score
    true_positives = correct_positives
    false_positives = total_negatives - correct_negatives  
    false_negatives = total_positives - true_positives  

    precision_positives = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall_positives = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score_positives = 2 * (precision_positives * recall_positives) / (precision_positives + recall_positives) if (precision_positives + recall_positives) > 0 else 0

    # Storing F1 scores for plotting
    train_f1_1_scores.append(f1_score_positives) 

    print(f'Train F1 Score: {f1_score_positives:.4f}')

    # Validation loop
    model.eval()  
    val_loss = 0
    val_correct_predictions = 0
    val_total_predictions = 0
    val_correct_positives = 0
    val_total_positives = 0
    val_correct_negatives = 0
    val_total_negatives = 0

    with torch.no_grad(): 
        for batched_graph, labels in val_dataloader:
            batched_graph = batched_graph.to(TORCH_DEVICE)
            labels = labels.to(TORCH_DEVICE).float()

            logits = model(batched_graph, batched_graph.ndata['feat']).squeeze()
            loss = loss_fn(logits, labels)

            val_loss += loss.item()

            # Convert logits to probabilities with sigmoid 
            preds = (torch.sigmoid(logits) >= 0.5).float()

            # Convert probabilities to binary predictions
            val_correct_predictions += (preds == labels).float().sum()

            val_total_predictions += labels.size(0)

            # Calculating correct positives and negatives
            val_correct_positives += ((preds == 1) & (labels == 1)).float().sum().item()
            val_total_positives += (labels == 1).float().sum().item()
            val_correct_negatives += ((preds == 0) & (labels == 0)).float().sum().item()
            val_total_negatives += (labels == 0).float().sum().item()

    val_avg_loss = val_loss / len(val_dataloader)
    val_accuracy = val_correct_predictions / val_total_predictions
    val_positive_accuracy = val_correct_positives / val_total_positives if val_total_positives > 0 else 0
    val_negative_accuracy = val_correct_negatives / val_total_negatives if val_total_negatives > 0 else 0

    print(f'Validation - Loss: {val_avg_loss:.4f}, Accuracy: {val_accuracy:.4f}')
    print(f'Validation - Correct 1\'s (Positives) Accuracy: {val_positive_accuracy:.4f}')
    print(f'Validation - Correct 0\'s (Negatives) Accuracy: {val_negative_accuracy:.4f}')

    # Calculating Precision, Recall, and F1 score
    val_true_positives = val_correct_positives
    val_false_positives = val_total_negatives - val_correct_negatives  
    val_false_negatives = val_total_positives - val_true_positives  

    val_precision_positives = val_true_positives / (val_true_positives + val_false_positives) if (val_true_positives + val_false_positives) > 0 else 0
    val_recall_positives = val_true_positives / (val_true_positives + val_false_negatives) if (val_true_positives + val_false_negatives) > 0 else 0
    val_f1_score_positives = 2 * (val_precision_positives * val_recall_positives) / (val_precision_positives + val_recall_positives) if (val_precision_positives + val_recall_positives) > 0 else 0

    # Storing F1 scores for plotting
    val_f1_1_scores.append(val_f1_score_positives) 

    print(f'Validation F1 Score: {val_f1_score_positives:.4f}')

    # Saving model with best loss value
    if val_avg_loss < best_val_loss:
        best_val_loss = val_avg_loss

        # Save the best model
        torch.save(model.state_dict(), best_model_path)  
        print(f"Epoch {epoch+1}: New best model saved with loss {best_val_loss:.4f}")

print("\nTraining completed!\n")

model.load_state_dict(torch.load(best_model_path))
print("Loaded the best model for testing.\n")

# Test the best trained model
model.eval()  
test_loss = 0
test_correct_predictions = 0
test_total_predictions = 0
test_correct_positives = 0
test_total_positives = 0
test_correct_negatives = 0
test_total_negatives = 0

with torch.no_grad(): 
    for batched_graph, labels in test_dataloader:
        batched_graph = batched_graph.to(TORCH_DEVICE)
        labels = labels.to(TORCH_DEVICE).float()

        logits = model(batched_graph, batched_graph.ndata['feat']).squeeze()
        loss = loss_fn(logits, labels)
        test_loss += loss.item()

        # Convert logits to probabilities with sigmoid
        preds = (torch.sigmoid(logits) >= 0.5).float()

        # Convert probabilities to binary predictions
        test_correct_predictions += (preds == labels).float().sum()

        test_total_predictions += labels.size(0)

        # Calculating correct positives and negatives
        test_correct_positives += ((preds == 1) & (labels == 1)).float().sum().item()
        test_total_positives += (labels == 1).float().sum().item()
        test_correct_negatives += ((preds == 0) & (labels == 0)).float().sum().item()
        test_total_negatives += (labels == 0).float().sum().item()

test_avg_loss = test_loss / len(test_dataloader)
test_accuracy = test_correct_predictions / test_total_predictions
test_positive_accuracy = test_correct_positives / test_total_positives if test_total_positives > 0 else 0
test_negative_accuracy = test_correct_negatives / test_total_negatives if test_total_negatives > 0 else 0

print(f'Test - Loss: {test_avg_loss:.4f}, Accuracy: {test_accuracy:.4f}')
print(f'Test - Correct 1\'s (Positives) Accuracy: {test_positive_accuracy:.4f}')
print(f'Test - Correct 0\'s (Negatives) Accuracy: {test_negative_accuracy:.4f}')

# Calculating Precision, Recall, and F1 score
test_true_positives = test_correct_positives
test_false_positives = test_total_negatives - test_correct_negatives  
test_false_negatives = test_total_positives - test_true_positives 

test_precision_positives = test_true_positives / (test_true_positives + test_false_positives) if (test_true_positives + test_false_positives) > 0 else 0
test_recall_positives = test_true_positives / (test_true_positives + test_false_negatives) if (test_true_positives + test_false_negatives) > 0 else 0
test_f1_score_positives = 2 * (test_precision_positives * test_recall_positives) / (test_precision_positives + test_recall_positives) if (test_precision_positives + test_recall_positives) > 0 else 0

print(f'Test F1 Score: {test_f1_score_positives:.4f}')

print("\nOutputting F1 score plot...")

# plotting the F1 scores curve during training and validation
epochs = range(1, num_epochs + 1)

# Blue line for training F1 scores
plt.plot(epochs, train_f1_1_scores, 'b-', label='Training') 
# Red line for validation F1 scores
plt.plot(epochs, val_f1_1_scores, 'r-', label='Validation')  

plt.title('Training and Validation F1 Scores')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.show()