import os
from glob import glob
import numpy as np

def find_files_with_string(directory, target_string):
    matching_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if target_string in file:
                matching_files.append(file)
                
    # print(matching_files)
    return matching_files
def sort_file_names_in_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    sorted_lines = sorted(lines,key=lambda x: x.lower())

    with open(file_path, 'w') as file:
        file.writelines(sorted_lines)



directory_path = '/home/lwh/code/shap-e/results/OT/cvx_1e-05/invx_0.0/lr_0.0001/act_celu/quad_False/layer_8/try_0/mesh'  
target_string = "_text_"  


matching_files = find_files_with_string(directory_path, target_string)


file_path = 'cvx_1_invx_0_part_text_name.txt' 
with open(file_path, 'w') as fp:
    for filename in matching_files:
        fp.write(filename + '\r\n')
  
sort_file_names_in_file(file_path)

with open(file_path, 'w') as fp:
    for filename in matching_files:
        # fp.write(directory_path+filename + '\r\n')
        fp.write(filename + '\r\n')


print('[*] Have done!')




