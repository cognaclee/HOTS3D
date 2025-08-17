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
import pickle

from shap_e.models.generation.pretrained_clip import FrozenImageCLIP
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config


os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used
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

        model_list = model_list + png_list
        text_list = text_list + [text]*len(png_list)
        name_list = name_list+ [model_name]*len(png_list)

        
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

    #clip_model = FrozenImageCLIP(device,clip_name= "ViT-B/32")#ViT-B/32  /  ViT-L/14
    clip_model = FrozenImageCLIP(device,clip_name= "ViT-L/14")
    batch_size = 140
    '''
    dataroot='./datasets'
    phase = 'test'
    max_dataset_size = 3400#12000 3455
    img_list, text_list, name_list = text2ShapeDataset(dataroot, 
                                            phase=phase, cat='all',max_dataset_size=max_dataset_size)
    '''
    text_file = './datasets/text2shape/test/text.txt'  
    text_list = []
    orders = []
    with open(text_file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n') 
            oneline = line.split('_')
            one_text = [oneline[1]]*20
            text_list = text_list+one_text
            orders.append(oneline[0])
            #text_list.append(line)
    print('text_list=',len(text_list))        

    image_dir = './results/text2shape/png'
    img_list = []
    for idx in orders:
        subdir = os.path.join(image_dir,str(idx))
        img20 = glob(subdir+'/*.'+'png')
        img_list = img_list+img20

    print('img_list=',len(img_list))

    rem = len(img_list)%batch_size
    if rem !=0:
        img_list = img_list[:-rem]
        text_list = text_list[:-rem]
        # name_list = name_list[:-rem]
    
    print('[*] %d sample pairs loaded.' % (len(img_list)))
    print('[*] %d texts loaded.' % (len(text_list)))
    #"""
    
    clip_result = []
    data_len = len(img_list)
    
    #gt_idx = np.array(range(batch_size),dtype=np.int32)//20
    gt_idx = np.array(range(batch_size),dtype=np.int32)
    print('gt_idx.shape=',gt_idx.shape)

    clip_r_precious = 0
    all_samples = 0
    for i in range(0,data_len,batch_size):
        prompts = text_list[i:i+batch_size]
        img_names = img_list[i:i+batch_size]
        imgs = []
        #j=0
        for imgname in img_names:
            img = load_image(imgname)
            imgs.append(img)
        
        with torch.no_grad():
            text_features = clip_model(batch_size=batch_size, texts=prompts)
            image_features = clip_model(batch_size=batch_size, images=imgs)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = clip_model.model.clip_model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()

            #clip_prediction = torch.argsort(logits_per_image, dim=1, descending=True)[0, 0].item()
            clip_prediction = torch.argsort(logits_per_image, dim=1, descending=True)
            
            # k = 5,  clip_r_precious = 0.3325
            flag = (gt_idx<0)
            for k in range(80):
                pred = clip_prediction[:,k].cpu().numpy()
                #pred = clip_prediction[:,k].cpu().numpy()//20
                #pred = pred.astype(np.int32)
                flag = (flag | (pred==gt_idx))
            one_precious = np.sum(flag)
            clip_r_precious += one_precious
            print('*** Batch clip_r_precious = ',one_precious/batch_size)
            all_samples += batch_size

        new_entry = (prompts, img_names, clip_prediction)
        clip_result.append(new_entry)
    
    clip_r_precious = 1.0*clip_r_precious/all_samples
    print('clip_r_precious=',clip_r_precious)
    #"""

