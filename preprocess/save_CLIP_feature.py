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
import pandas as pd
import argparse

from shap_e.models.generation.pretrained_clip import FrozenImageCLIP
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config


os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"  # specify which GPU(s) to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path: str):
    with bf.BlobFile(image_path, "rb") as thefile:
        img = Image.open(thefile)
        img.load()
    return img

def text2ShapeDataset(dataroot, phase='train', cmax_dataset_size=None):
    data_folders = os.path.join(dataroot, phase)

    name_list = []
    text_list = []
    text_file = os.path.join(data_folders, 'text.txt')
    with open(text_file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n') 
            text_list.append(line)

    name_file = os.path.join(data_folders, 'name.txt')
    with open(name_file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n') 
            name_list.append(line)

    if max_dataset_size is not None:
        text_list = text_list[:max_dataset_size]
        name_list = name_list[:max_dataset_size]
    print('[*] %d samples loaded.' % (len(model_list)))

    return name_list, text_list 


def objaverseDataset(dataroot,max_dataset_size=None):
    csv_file = os.path.join(dataroot, 'Cap3D_automated_Objaverse.csv')
    data = pd.read_csv(csv_file)
    data = data.values
    print(data.shape)
    uids = data[:,0]
    texts = data[:,1]
    
    uid_text = {} 
    for i in range(len(uids)):
        uid = uids[i]
        uid_text[uid] = texts[i]
        
    
    model_list = []
    text_list = []

    subfolders = os.listdir(dataroot)
    subfolders = [f.name for f in os.scandir(dataroot) if f.is_dir()]
    for path in subfolders:
        sub_dir = os.path.join(dataroot, path)
        for onefile in os.listdir(sub_dir):
            try:
                text = uid_text[onefile]
                filename = os.path.join(sub_dir,onefile) 
                model_list.append(filename)
                text_list.append(text)
            except KeyError:
                print( onefile, " does not exist")


    if max_dataset_size is not None:
        model_list = model_list[:max_dataset_size]
        text_list = text_list[:max_dataset_size]
    print('[*] %d samples loaded.' % (len(model_list)))

    return model_list, text_list




parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['text2shape', 'objaverse'], default='text2shape', help='which dataset to use')
parser.add_argument('--dataroot', type=str, default="/mnt/data2/text2shape/")
parser.add_argument('--save_dir', type=str, default="/mnt/data2/text2shape/features/")
parser.add_argument('--batch_size', type=int, default=400)
parser.add_argument('--max_dataset_size', type=int, default=None)
parser.add_argument('--num_trial', type=int, default=1, help='Index of trial')
FLAGS = parser.parse_args()

###################################################
if __name__ == '__main__':

    clip_model = FrozenImageCLIP(device)
    
    #dataroot='./datasets/text2shape'
    dataroot='./datasets/text2shape/Cap3D/misc/RenderedImage_zips/'
    batch_size = FLAGS.batch_size
    phase = 'train'#'test'
    max_dataset_size = FLAGS.max_dataset_size
    if FLAGS.dataset == 'text2shape'
        train_img_list, train_text_list = text2ShapeDataset(FLAGS.dataroot, 'train',max_dataset_size)
        test_img_list, test_text_list = text2ShapeDataset(FLAGS.dataroot, 'test',max_dataset_size)
    else:
        img_list, text_list = objaverseDataset(FLAGS.dataroot, max_dataset_size)
        
    rem = len(img_list)%(batch_size*10)
    if rem !=0:
        img_list = img_list[:-rem]
        text_list = text_list[:-rem]
    
    if FLAGS.dataset == 'objaverse'
        ratio = 10
        train_img_list = []
        train_text_list = []
        test_img_list = []
        test_text_list = []
        for i in range(0,len(img_list),ratio):
            test_img_list.append(img_list[i])
            test_text_list.append(text_list[i])
            for j in range(1,ratio):
                train_img_list.append(img_list[i+j])
                train_text_list.append(text_list[i+j])
    
    
    print('[*] %d training sample pairs loaded.' % (len(train_img_list)))
    print('[*] %d test sample pairs loaded.' % (len(test_img_list)))
    
    #"""
    save_dir = os.path.join(FLAGS.save_dir,'train')  
    
    text_save_dir = os.path.join(save_dir,'text')
    if not os.path.exists(text_save_dir):
        os.makedirs(text_save_dir)
    image_save_dir = os.path.join(save_dir,'image')
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)  
             
    train_name_file = os.path.join(save_dir,'name.txt')
    with open(train_name_file, 'w') as fp:
        for name in train_img_list:
            fp.write(name+'\r\n')

    train_text_file = os.path.join(save_dir,'text.txt')
    with open(train_text_file, 'w') as fp:
        for text in train_text_list:
            fp.write(text+'\r\n') 

    
    data_len = len(train_img_list)
    for i in range(0,data_len,batch_size):
        prompts = train_text_list[i:i+batch_size]
        img_names = train_img_list[i:i+batch_size]
        imgs = []
        for imgname in img_names:
            imgs.append(load_image(imgname))
        
        
        text_latents = clip_model(batch_size=batch_size, texts=prompts)
        img_latents = clip_model(batch_size=batch_size, images=imgs)
        #img_latents = clip_model.embed_images_grid(imgs)#NDL

        feature_save_path = os.path.join(text_save_dir,'txt_'+str(i)+'.pt')
        torch.save(text_latents.cpu(), feature_save_path)
        feature_save_path = os.path.join(image_save_dir,'img_'+str(i)+'.pt')
        torch.save(img_latents.cpu(), feature_save_path)
    print('1. Training dataset have been done! It has ',data_len, ' files')
    
    #"""
    save_dir = os.path.join(FLAGS.save_dir,'test') 
    text_save_dir = os.path.join(save_dir,'text')
    if not os.path.exists(text_save_dir):
        os.makedirs(text_save_dir)
    image_save_dir = os.path.join(save_dir,'image')
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)  
              
    test_name_file = os.path.join(save_dir,'name.txt')
    with open(test_name_file, 'w') as fp:
        for name in test_img_list:
            fp.write(name+'\r\n')

    test_text_file = os.path.join(save_dir,'text.txt')
    with open(test_text_file, 'w') as fp:
        for text in test_text_list:
            fp.write(text+'\r\n') 
    
    
    print('Test dataset file name and text prompt have been written!') 
    data_len = len(test_img_list)

    for i in range(0,data_len,batch_size):
        prompts = test_text_list[i:i+batch_size]
        img_names = test_img_list[i:i+batch_size]
        imgs = []
        for imgname in img_names:
            #print('imgname=',imgname)
            imgs.append(load_image(imgname))
        
        
        text_latents = clip_model(batch_size=batch_size, texts=prompts)
        img_latents = clip_model(batch_size=batch_size, images=imgs)
        #img_latents = clip_model.embed_images_grid(imgs)#NDL

        feature_save_path = os.path.join(text_save_dir,'txt_'+str(i)+'.pt')
        torch.save(text_latents.cpu(), feature_save_path)
        feature_save_path = os.path.join(image_save_dir,'img_'+str(i)+'.pt')
        torch.save(img_latents.cpu(), feature_save_path)
    print('2. Test dataset have been done! It has ',data_len, ' files')
        
