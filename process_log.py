'''
This code analysize the log.txt file to display or extract relavent information 
- Display a circuit assessment and feature extraction time vs circuit size figure
- Display max, mean, min information on areas such as: assessment times, total nodes, estimated/actual leaks, accuracies, number of false negatives, etc.
- Display a accuracies distribution figure
'''

import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import linregress
import os

# Function to process the log file and extract information into lists
def process_log_file(file_path):
    file_name = []
    model_assessment_times = []
    total_nodes = []
    estimated_leaks = []
    actual_leaks = []
    accuracies_1 = []
    accuracies_0 = []
    accuracies = []
    neg = []
    actual_leak = 0
    
    with open(file_path, 'r') as file:
        lines = file.readlines()

        problem_file = 0
        
        for line in lines:
            if line.startswith('File name:'):
                filename = line.split(': ')[1].strip()
                file_name.append(filename)
            elif line.startswith('Model assessment time:'):
                model_assessment_times.append(float(line.split(': ')[1].strip()))
            elif line.startswith('Feature extraction time:'):
                number_str = line.split(': ')[1].strip()
                number_str = number_str.strip('[]')
                feature_times.append(float(number_str))
            elif line.startswith('Total nodes in circuit:'):
                total_nodes.append(int(line.split(': ')[1].strip()))
            elif line.startswith('Node leaking key information (Estimated):'):
                estimated_leaks.append(int(line.split(': ')[1].strip()))
            elif line.startswith('Node leaking key information (Actual):'):
                actual_leak = int(line.split(': ')[1].strip())
                actual_leaks.append(actual_leak)
            elif line.startswith('Pos class accuracy:'):
                accuracies_1.append(float(line.split(': ')[1].strip()))
            elif line.startswith('Neg class accuracy:'):
                accuracies_0.append(float(line.split(': ')[1].strip()))
            elif line.startswith('Overall accuracy:'):
                accuracies.append(float(line.split(': ')[1].strip()))
            elif line.startswith('Number of false negatives:'):
                neg_temp = float(line.split(': ')[1].strip())
                neg.append(neg_temp)

                if neg_temp > 10:
                    print(f"file name: {filename} amount of leakage: {actual_leak} false negatives: {neg_temp}")
                    problem_file += 1
        
        print(f"Amount of files to check: {problem_file}")
                
    return model_assessment_times, total_nodes, estimated_leaks, actual_leaks, accuracies_1, accuracies_0, accuracies, neg

# Function to calculate max, mean, and min for a given list
def calculate_max_mean_min(values):
    max_val = max(values)
    mean_val = sum(values) / len(values)
    min_val = min(values)
    return max_val, mean_val, min_val

# Specify the path to your log file
current_path = os.getcwd()
file_path = os.path.join(current_path, 'test_comp_wBest_model15_log.txt')

# Process the log file
model_assessment_times, total_nodes, estimated_leaks, actual_leaks, accuracies_1, accuracies_0, accuracies, neg = process_log_file(file_path)

# Plotting Model Assessment Time vs. Total Nodes in Circuit
mpl.rcParams.update({'font.size': 12})  # You can adjust the size as needed
plt.figure(figsize=(10, 6))
plt.scatter(total_nodes, model_assessment_times, color='blue', marker='o')
plt.xlabel('Total Nodes in Circuit')
plt.ylabel('Model Assessment Time (seconds)')

# UNCOMMENT to save figure
#file_path = os.path.join(current_path, 'time_vs_size_wbest15.pdf')
#plt.savefig(file_path)
plt.show()

# Perform linear regression and calculate R-squared
slope, intercept, r_value, p_value, std_err = linregress(total_nodes, model_assessment_times)
r_squared = r_value**2

print("Slope:", slope)
print("Intercept:", intercept)
print("R-squared:", r_squared)

# Calculating and printing the stats for each list
stats = {
    #"Feature Extraction Times": calculate_max_mean_min(feature_times),
    "Model Assessment Times": calculate_max_mean_min(model_assessment_times),
    "Total Nodes": calculate_max_mean_min(total_nodes),
    "Pos Accuracies": calculate_max_mean_min(accuracies_1),
    "Neg Accuracies": calculate_max_mean_min(accuracies_0),
    "Accuracies": calculate_max_mean_min(accuracies),
    "False Negitive": calculate_max_mean_min(neg),
}

print("\n")

for category, (max_val, mean_val, min_val) in stats.items():
    print(f"{category} - Max: {max_val}, Mean: {mean_val:.4f}, Min: {min_val}")

# Plotting the distribution of accuracies
mpl.rcParams.update({'font.size': 12})  # You can adjust the size as needed
plt.figure(figsize=(8, 6))
plt.hist(accuracies, bins=20, color='palegreen', edgecolor='darkgreen')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')

# UNCOMMENT to save figure
#file_path = os.path.join(current_path, 'accuracy_distribution.pdf')
#plt.savefig(file_path, bbox_inches='tight')
plt.show()

