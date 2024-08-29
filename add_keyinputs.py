'''
This program adds Keyinput to a .bench circuit
- The program takes and process all .bench file from a input directory
- The program assume the input file does not inherantly contain keyinputs
- The percentage of keyinputs can be adjusted, otherwise default to 30-80 percent selected randomly
- The program will output processed .bench circuits, with keyinputs added, to the selected output directory
- Output naming ex. 'keyed_output_[index].bench'
'''

import random
import re
import glob
import os
import time 

# Set the RNG seed using the current system time 
# to ensure that the data would not repeat if augumented more than once
print(time.time())
random.seed(time.time())

# Add keyinput to the selected file and output file with keyinput added
def add_keyinputs_to_bench_file(file_path, output_file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Identify all input lines
    input_lines = [line for line in lines if line.startswith('INPUT')]
    total_inputs = len(input_lines)

    # Randomly a percentage defined by the range below 
    # ex. 30-80 percent of input change to keyinputs
    percentage = random.randint(30, 80) / 100
    print(f"Selected percentage: {percentage * 100}%")

    # Calculate the number of inputs to be designated as 'keyinputs'
    keyinputs_count = round(total_inputs * percentage)
    print(f"Total inputs: {total_inputs}, Keyinputs needed: {keyinputs_count}")

    # Randomly select input nodes to be renamed
    selected_input_lines = random.sample(input_lines, keyinputs_count)
    selected_input_names = [line.split('(')[1].split(')')[0] for line in selected_input_lines]

    # Prepare a mapping for renaming
    # ensuring unique 'keyinput' names
    keyinput_mapping = {}
    for index, name in enumerate(selected_input_names):
        new_name = f"keyinput{index}"
        keyinput_mapping[name] = new_name

    # Rename selected input nodes 
    # ensuring unique replacements
    new_lines = []
    for line in lines:
        for original_name, new_name in keyinput_mapping.items():
            # Ensure exact match to avoid partial replacements
            line = re.sub(r'\b' + re.escape(original_name) + r'\b', new_name, line)
        new_lines.append(line)

    # Write to output file
    with open(output_file_path, 'w') as file:
        file.writelines(new_lines)

    print(f"File processed and saved as '{output_file_path}' with {keyinputs_count} keyinputs.")

# Add keyinputs to all file to defined iterations
def process_directory(input_directory, output_directory, start_index):

    bench_files = glob.glob(os.path.join(input_directory, '*.bench'))
    os.makedirs(output_directory, exist_ok=True)  # Ensure output directory exists

    current_index = start_index

    # Repeat the process by n times based on the selected value
    # ex. repeat 3 times
    for iteration in range(3):  
        for file_path in bench_files:
            base_name = os.path.basename(file_path)

            #change the output file name if desired
            output_file_path = os.path.join(output_directory, f'keyed_output_{current_index}.bench')
            add_keyinputs_to_bench_file(file_path, output_file_path)

            current_index += 1  

    print(f"Finished processing. Files are saved in '{output_directory}'.")

# replace with your input and output directories here
input_directory = '../path_to_your_input_dir'
output_directory = '../path_to_your_output_dir'

# Adjust as needed
# ex. starting at index 1
start_index = 1  

# Example usage
process_directory(input_directory, output_directory, start_index)