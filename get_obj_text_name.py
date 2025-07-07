import os
from glob import glob
import numpy as np

def find_and_save_matching_lines(input_file_path, search_file_path, output_file_path):
    
    with open(search_file_path, 'r') as search_file:
        search_lines = {line[:10] for line in search_file}

    
    matched_lines = set()

   
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            first_100_chars = line[:10]
            if first_100_chars in search_lines and first_100_chars not in matched_lines:
                output_file.write(line)
                matched_lines.add(first_100_chars)

# file path
input_file_path = '/home/lwh/code/shap-e/datasets/ShapeNet/feature/test/text.txt'  # 
search_file_path = '/home/lwh/code/shap-e/cvx_1_invx_0_part_text_name.txt' #
output_file_path ='cvx_1_invx_0_total_text.txt'  # 

find_and_save_matching_lines(input_file_path, search_file_path, output_file_path)

print('[*] Have done!', output_file_path)
