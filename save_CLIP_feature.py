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



###################################################
if __name__ == '__main__':

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    '''
    text_model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    img_model = load_model('image300M', device=device)
    '''
    clip_model = FrozenImageCLIP(device)
    

    #dataroot='/home/lwh/code/shap-e/datasets'
    dataroot='/mnt/data2/Cap3D/misc/RenderedImage_zips/'
    batch_size = 400
    phase = 'test'#'test'
    max_dataset_size = None#6000#600000#3600
    #img_list, text_list, name_list = text2ShapeDataset(dataroot, phase=phase, cat='all',max_dataset_size=max_dataset_size)
    img_list, text_list = objaverseDataset(dataroot, max_dataset_size=1200000)
    rem = len(img_list)%(batch_size*10)
    if rem !=0:
        img_list = img_list[:-rem]
        text_list = text_list[:-rem]
    
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
    
    
    print('[*] %d sample pairs loaded.' % (len(img_list)))
    print('[*] %d texts loaded.' % (len(text_list)))
    
    """
    save_dir = '/mnt/data2/Objaverse/train/'  
    
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
        print('i=',i)
        #print('text_latents.shape=',text_latents.shape)
        img_latents = clip_model(batch_size=batch_size, images=imgs)
        #img_latents = clip_model.embed_images_grid(imgs)#NDL

        feature_save_path = os.path.join(text_save_dir,'txt_'+str(i)+'.pt')
        torch.save(text_latents.cpu(), feature_save_path)
        feature_save_path = os.path.join(image_save_dir,'img_'+str(i)+'.pt')
        torch.save(img_latents.cpu(), feature_save_path)
    print('1. Training dataset have been done! It has ',data_len, ' files')
    
    """
    save_dir = '/mnt/data2/Objaverse/test/' 
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
        print('i=',i)
        #print('text_latents.shape=',text_latents.shape)
        img_latents = clip_model(batch_size=batch_size, images=imgs)
        #img_latents = clip_model.embed_images_grid(imgs)#NDL

        feature_save_path = os.path.join(text_save_dir,'txt_'+str(i)+'.pt')
        torch.save(text_latents.cpu(), feature_save_path)
        feature_save_path = os.path.join(image_save_dir,'img_'+str(i)+'.pt')
        torch.save(img_latents.cpu(), feature_save_path)
    print('2. Test dataset have been done! It has ',data_len, ' files')
    
    
    
    
    
    
    """
    rem = len(img_list)%batch_size
    if rem !=0:
        img_list = img_list[:-rem]
        text_list = text_list[:-rem]
        name_list = name_list[:-rem]
    
    save_dir = './datasets/ShapeNet/feature/'
    text_save_dir = os.path.join(save_dir,phase,'text')
    if not os.path.exists(text_save_dir):
        os.makedirs(text_save_dir)
    image_save_dir = os.path.join(save_dir,phase,'image')
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)
    
    print('[*] %d sample pairs loaded.' % (len(img_list)))
    print('[*] %d texts loaded.' % (len(text_list)))
    save_text_file = os.path.join(save_dir,phase,'text.txt')
    with open(save_text_file, 'w') as fp:
        for text in text_list:
            new_text = text.replace("\n", " ")
            fp.write(new_text+'\r\n')
    
    save_name_file = os.path.join(save_dir,phase,'name.txt')
    with open(save_name_file, 'w') as fp:
        for name in img_list:
            fp.write(name+'\r\n')
    
    text_guidance_scale = 15.0
    img_guidance_scale = 3.0
    data_len = len(img_list)

    """
    """
    for i in range(0,data_len,batch_size):
        prompts = text_list[i:i+batch_size]
        img_names = img_list[i:i+batch_size]
        imgs = []
        j=0
        for imgname in img_names:
            img = load_image(imgname)
            imgs.append(img)
            #j+=1
            '''
            if j==107 or j==108:
                img.show()
                print('imgname = ', imgname)
                print('text = ', text_list[i+j-1])
                
                time.sleep(20)
            '''
        
        text_latents = clip_model(batch_size=batch_size, texts=prompts)
        print('i=',i)
        #print('text_latents.shape=',text_latents.shape)
        img_latents = clip_model(batch_size=batch_size, images=imgs)
        #img_latents = clip_model.embed_images_grid(imgs)#NDL

        feature_save_path = os.path.join(text_save_dir,'txt_'+str(i)+'.pt')
        torch.save(text_latents.cpu(), feature_save_path)
        feature_save_path = os.path.join(image_save_dir,'img_'+str(i)+'.pt')
        torch.save(img_latents.cpu(), feature_save_path)
        
    """
        
