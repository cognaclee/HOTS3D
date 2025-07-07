import os
from glob import glob
import numpy as np

external_folder = '/home/lwh/code/PureCLIPNeRF/logs/shapenet/png' 

img_list = []
max_idx = 500
for idx in range(max_idx):
    sub_dir = os.path.join(external_folder, str(idx))
    if os.path.isdir(sub_dir):
        png_list = glob(sub_dir+'/*.'+'png')
        # png_ch = np.random.choice(png_list, 1, replace=False).tolist()
        #print('png_ch=',png_ch)
        img_list = img_list + png_list


text_file = '/home/lwh/code/shap-e/PureCLIPNeRF50.txt'
all_text_list = []
with open(text_file, 'r') as f:
    for line in f.readlines():
        line = line.strip('\n')
        all_text_list.append(line)
     

select_text_list = []       
for i in range(max_idx):
    idx = i*20
    #print('img_list[idx]=',img_list[idx])
    baseName = img_list[idx].split('/')[-1]
    baseName = baseName.split('_')[0]
    print('***baseName=',baseName)
    for j in range(len(all_text_list)):
        if baseName == all_text_list[j][:240]:
            #select_text_list.append(all_text_list[j])
            select_text_list = select_text_list + [all_text_list[j]]*20
            #print('%%% all_text_list[j]=',all_text_list[j])
            break
        elif baseName[:-5] == all_text_list[j][:240]:
            #select_text_list.append(all_text_list[j])
            select_text_list = select_text_list + [all_text_list[j]]*20
            print('111 all_text_list[j]=',all_text_list[j])
            break
        elif baseName[:-6] == all_text_list[j][:240]:
            #select_text_list.append(all_text_list[j])
            select_text_list = select_text_list + [all_text_list[j]]*20
            print('222 all_text_list[j]=',all_text_list[j])
            break
    

save_name_file = 'PureCLIPNeRF_PNG_name.txt'
with open(save_name_file, 'w') as fp:
    for name in img_list:
        fp.write(name+'\r\n')

save_name_file = 'PureCLIPNeRF_text.txt'
with open(save_name_file, 'w') as fp:
    for text in select_text_list:
        fp.write(text+'\r\n')

print('[*] Have done!')


