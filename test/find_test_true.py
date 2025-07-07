import torch
import os
import json
import csv
from glob import glob
from PIL import Image
import blobfile as bf
from termcolor import cprint
from tqdm import tqdm
import numpy as np
import time

from shap_e.models.generation.pretrained_clip import FrozenImageCLIP
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config


os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"  # specify which GPU(s) to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path: str):
    with bf.BlobFile(image_path, "rb") as thefile:
        img = Image.open(thefile)
        img.load()
    return img



def text2ShapeDataset(dataroot, phase='train', cat='all',max_dataset_size=None):
    text_csv = f'{dataroot}/ShapeNet/text2shape/captions.tablechair_{phase}.csv'

    with open(text_csv) as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader, None)

        data = [row for row in reader]

    with open(f'{dataroot}/dataset_info_files/info-shapenet.json') as f:
        info = json.load(f)

    cat_to_id = info['cats']
    id_to_cat = {v: k for k, v in cat_to_id.items()}
        
    assert cat.lower() in ['all', 'chair', 'table']
    if cat == 'all':
        valid_cats = ['chair', 'table']
    else:
        valid_cats = [cat]
        
    model_list = []
    name_list = []
    text_list = []

    #for d in tqdm(data, total=len(data), desc=f'readinging text data from {text_csv}'):
    for d in tqdm(data, total=len(data)):
        id, model_id, text, cat_i, synset, subSynsetId = d
            
        if cat_i.lower() not in valid_cats:
            continue
            
        png_path = f'{dataroot}/ShapeNet/PNG/{synset}/{model_id}/'
        model_name = f'/{synset}/{model_id}'

        if not os.path.exists(png_path):
            continue
            # {'Chair': 26523, 'Table': 33765} vs {'Chair': 26471, 'Table': 33517}
            # not sure why there are some missing files
        else:
            png_list = glob(png_path+'/*.'+'png')
        
        if phase=='train':
            model_list = model_list + png_list
            text_list = text_list + [text]*len(png_list)
            name_list = name_list+ [model_name]*len(png_list)
        else:
            text_list = text_list + [text]*2
            name_list = name_list+ [model_name]*2
            #idx = np.random.randint(0, len(png_list), size = (2,),dtype=np.int64)
            png_ch = np.random.choice(png_list, 2, replace=False).tolist()
            #print('idx=',idx)
            #print('png_ch=',png_ch)
            #model_list = model_list + png_list[idx]
            model_list = model_list + png_ch
        
        if (max_dataset_size is not None) and (len(model_list)>max_dataset_size):
            break


    if max_dataset_size is not None:
        model_list = model_list[:max_dataset_size]
        text_list = text_list[:max_dataset_size]
        name_list = name_list[:max_dataset_size]
    print('[*] %d samples loaded.' % (len(model_list)))

    return model_list, text_list, name_list




###################################################
if __name__ == '__main__':

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    '''
    text_model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    img_model = load_model('image300M', device=device)
    '''
    clip_model = FrozenImageCLIP(device)
    

    dataroot='/home/lwh/code/shap-e/datasets'
    batch_size = 400
    phase = 'test'#'test'
    max_dataset_size = 3600#3600
    img_list, text_list, name_list = text2ShapeDataset(dataroot,phase=phase, cat='all',max_dataset_size=max_dataset_size)
   
   
    print("img_list：",img_list)
    # print("text_list:",text_list)


    # print("name_list:",name_list)

    # test_path='/home/lwh/code/shap-e/cvx0_invx1_PNG_text.txt'
    test_path='/home/lwh/code/shap-e/ab_OBJ/cv0_invx0_OBJ/text.txt'
    #
    with open(test_path, "r") as text_file:
        text_data = text_file.read().splitlines()

    print("text_data",text_data)

    # 
    test_true_PNG_save_path = "/home/lwh/code/shap-e/ab_OBJ/cv0_invx0_OBJ/cvx0_invx0_True_PNG_name.txt"
    # os.makedirs(test_data_path, exist_ok=True)
    
    save_dir = '/home/lwh/code/shap-e/datasets/ShapeNet/PNG'

    # written_texts = set()
    
    with open(test_true_PNG_save_path, "w") as fp:
        for i, line in enumerate(text_data):#50
            for j, name in enumerate(text_list):#3600
                if line == name:
                # if name in line :
                    for m in range(0,20):
                        fp.write(save_dir+name_list[j]+'/'+str(m)+'.png'+'\r\n')
                    break


    print("have saved data")



    rem = len(img_list)%batch_size
    if rem !=0:
        img_list = img_list[:-rem]
        text_list = text_list[:-rem]
        name_list = name_list[:-rem]
    

    
    # save_dir = './datasets/ShapeNet/feature/'
    # text_save_dir = os.path.join(save_dir,phase,'text')
    # if not os.path.exists(text_save_dir):
    #     os.makedirs(text_save_dir)
    # image_save_dir = os.path.join(save_dir,phase,'image')
    # if not os.path.exists(image_save_dir):
    #     os.makedirs(image_save_dir)
    
    # print('[*] %d sample pairs loaded.' % (len(img_list)))
    # print('[*] %d texts loaded.' % (len(text_list)))
    # save_text_file = os.path.join(save_dir,phase,'text.txt')
    # with open(save_text_file, 'w') as fp:
    #     for text in text_list:
    #         new_text = text.replace("\n", " ")
    #         fp.write(new_text+'\r\n')
    
    # save_name_file = os.path.join(save_dir,phase,'name.txt')
    # with open(save_name_file, 'w') as fp:
    #     for name in img_list:
    #         fp.write(name+'\r\n')
    
    # text_guidance_scale = 15.0
    # img_guidance_scale = 3.0
    # data_len = len(img_list)

    # #"""
    # for i in range(0,data_len,batch_size):
    #     prompts = text_list[i:i+batch_size]
    #     img_names = img_list[i:i+batch_size]
    #     imgs = []
    #     j=0
    #     for imgname in img_names:
    #         img = load_image(imgname)
    #         imgs.append(img)
    #         #j+=1
    #         '''
    #         if j==107 or j==108:
    #             img.show()
    #             print('imgname = ', imgname)
    #             print('text = ', text_list[i+j-1])
                
    #             time.sleep(20)
    #         '''
    #     #'''
    #     text_latents = clip_model(batch_size=batch_size, texts=prompts)
    #     print('i=',i)
    #     #print('text_latents.shape=',text_latents.shape)
    #     img_latents = clip_model(batch_size=batch_size, images=imgs)
    #     #img_latents = clip_model.embed_images_grid(imgs)#NDL

    #     feature_save_path = os.path.join(text_save_dir,'txt_'+str(i)+'.pt')
    #     torch.save(text_latents.cpu(), feature_save_path)
    #     feature_save_path = os.path.join(image_save_dir,'img_'+str(i)+'.pt')
    #     torch.save(img_latents.cpu(), feature_save_path)
    #     #'''
    # #"""
        
