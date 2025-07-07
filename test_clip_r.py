"""
import os
from glob import glob
import torch
import numpy as np
if __name__ == '__main__':

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
 
    text_file = '/home/lwh/code/LFD/test_400.txt'
    text_list = []
    with open(text_file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n') 
            #one_text = [line]*20
            #text_list = text_list+one_text
            text_list.append(line)
    print('text_list=',len(text_list))        

 
    #img_list = glob('./3DFuse_Picture_result'+'/*.'+'png') #图片路径换掉
    img_list = glob('/mnt/data2/results/taps3d2png/*/*.png', recursive=True)
    print('img_list=',len(img_list))
"""
"""
import os
from glob import glob

# 根目录路径
base_dir = '/mnt/data2/results/Michelangelo2png'

# 获取所有文件夹路径
folder_paths = [os.path.join(base_dir, folder) for folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, folder))]

# 检查每个文件夹的文件数量
missing_files = []
for folder in folder_paths:
    png_files = glob(os.path.join(folder, '*.png'))
    if len(png_files) != 20:
        missing_files.append((folder, len(png_files)))

# 输出不符合预期的文件夹
if missing_files:
    for folder, count in missing_files:
        print(f"文件夹 {folder} 中的 .png 文件数量为 {count}，不是 20")
else:
    print("所有文件夹中都包含 20 张 .png 文件")
"""
import os
from glob import glob
#img_list = glob('/mnt/data2/results/taps3d2png'+'/*.'+'png')
img_list = glob('/mnt/data2/results/taps3d2png/*/*.png', recursive=True)
print('img_list=',len(img_list))
for img_path in img_list[:50]:
    print(img_path)
