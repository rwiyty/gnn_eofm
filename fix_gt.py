'''
This program comments out unncessary information on the ground truth files
- The program work in tendom with the raw output created by runing 'autorun_cus.sh' with the adjoining_gate program (credit to Thomas Wojtal)
- The output contains unnecessary information which might confuse the parser when using/training the GNN_EOFM model
- If only node related output are taken this program may not be necessary 
- The program simply comments out unnecessary information on the ground truth files in specified directory
'''

import glob
import os

# Get rid of unnecessary information in raw output files in selected directory
def comment_out_lines(directory):

    # Loop through all .txt files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # Comment out lines that do not contain "="
            # effectively comment out all the description lines
            new_lines = ['#' + line if '=' not in line else line for line in lines]
            
            # Overwrite the file with modified lines
            with open(file_path, 'w') as file:
                file.writelines(new_lines)
            
            print(f"Processed {filename}")

# replace with path to directory containing your ground truth + features in .txt files
directory = '../path_to_your_gt_dir'

# Example usage
comment_out_lines(directory)

print("finished processing all files")