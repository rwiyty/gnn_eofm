'''
This program cleans up Keyinputs in a .bench circuit
- The program takes and process all .bench file from a input directory
- The program will clean up all avaliabe 'keyinput' and replace with 'raninput'
- The program will output processed .bench circuits, with keyinputs cleaned out, to the selected output directory 
- Output naming ex. 'cleaned[index].bench'
'''

import os
import glob

# Clean up keyinputs in selected file
def replace_keyinputs_in_file(file_path, new_file_path):
    '''
    function will replace all instence of 'keyinput' with 'raninput'
    ex. "keyinput123" -> "raninput123"
    Becaue the naming of the keyinputs should not contain repeats, thus the replace inputs can key the original index after the keyinput
    '''
    with open(file_path, 'r') as file:
        lines = file.readlines()
    new_lines = [line.replace('keyinput', 'raninput') if 'keyinput' in line else line for line in lines]
    with open(new_file_path, 'w') as file:
        file.writelines(new_lines)

# load file from selected directory and store the files without keyinputs in selected directory
def process_all_bench_files(source_directory, target_directory):

    # ensure the target directory exists
    os.makedirs(target_directory, exist_ok=True)
    bench_files = glob.glob(os.path.join(source_directory, '*.bench'))
    
    for i, file_path in enumerate(bench_files, start=1):

        # change the output file name if desired
        new_file_path = os.path.join(target_directory, f'cleaned{i}.bench')

        # Replace keyinputs in the current file and save the result to the new file path
        replace_keyinputs_in_file(file_path, new_file_path)
        print(f"Processed '{file_path}' and saved the cleaned version to '{new_file_path}'.")

# replace with your input and output directories here
source_directory = '../path_to_your_input_dir'
target_directory = '../path_to_your_output_dir'

# Example usage
process_all_bench_files(source_directory, target_directory)

print("finished processing all files")