import torch
import os
import json
import csv
from glob import glob
from PIL import Image
import blobfile as bf
from termcolor import cprint
from tqdm import tqdm
import pandas as pd

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config


os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(2)
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
    cats_list = []
    text_list = []

    #for d in tqdm(data, total=len(data), desc=f'readinging text data from {text_csv}'):
    for d in tqdm(data, total=len(data)):
        id, model_id, text, cat_i, synset, subSynsetId = d
            
        if cat_i.lower() not in valid_cats:
            continue
            
        png_path = f'{dataroot}/ShapeNet/PNG/{synset}/{model_id}/'

        if not os.path.exists(png_path):
            continue
            # {'Chair': 26523, 'Table': 33765} vs {'Chair': 26471, 'Table': 33517}
            # not sure why there are some missing files
        else:
            png_list = glob(png_path+'/*.'+'png')
                
        model_list = model_list + png_list
        text_list = text_list + [text]*len(png_list)
        cats_list = cats_list+ [synset]*len(png_list)


    if max_dataset_size is not None:
        model_list = model_list[:max_dataset_size]
        text_list = text_list[:max_dataset_size]
        cats_list = cats_list[:max_dataset_size]
    print('[*] %d samples loaded.' % (len(model_list)))

    return model_list, text_list, cats_list
    

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))

    img_model = load_model('image300M', device=device)

    batch_size = 100
    dataroot='/mnt/data2/Cap3D/misc/RenderedImage_zips/'
    #dataroot='./datasets'
    #img_list, text_list, _ = text2ShapeDataset(dataroot, phase='train', cat='all',max_dataset_size=None)
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
    
    text_guidance_scale = 15.0
    img_guidance_scale = 3.0
    
    #save_dir = './datasets/ShapeNet/feature/'
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
    #feat_dim = 1048576
    #text_features = torch.zeros((data_len,feat_dim))
    #img_features = torch.zeros((data_len,feat_dim))

    #for i in range(19592,data_len,batch_size):
    for i in range(100,data_len,batch_size):
        prompts = train_text_list[i:i+batch_size]
        img_names = train_img_list[i:i+batch_size]
        imgs = []
        for imgname in img_names:
            imgs.append(load_image(imgname))

        text_latents = sample_latents(batch_size=batch_size,model=text_model,diffusion=diffusion,
            guidance_scale=text_guidance_scale,model_kwargs=dict(texts=prompts),
            progress=False,clip_denoised=True,use_fp16=True,use_karras=True,karras_steps=64,
            sigma_min=1e-3,sigma_max=160,s_churn=0,)
        img_latents = sample_latents(batch_size=batch_size,model=img_model,diffusion=diffusion,
            guidance_scale=img_guidance_scale,model_kwargs=dict(images=imgs),
            progress=False,clip_denoised=True,use_fp16=True,use_karras=True,karras_steps=64,
            sigma_min=1e-3,sigma_max=160,s_churn=0,)

        #text_features[i:i+batch_size,:]=text_latents.cpu()
        #img_features[i:i+batch_size,:]=img_latents.cpu()
        feature_save_path = os.path.join(text_save_dir,'txt_'+str(i)+'.pt')
        torch.save(text_latents.cpu(), feature_save_path)
        feature_save_path = os.path.join(image_save_dir,'img_'+str(i)+'.pt')
        torch.save(img_latents.cpu(), feature_save_path)
        print('Start index is: ', i)
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
    #feat_dim = 1048576
    #text_features = torch.zeros((data_len,feat_dim))
    #img_features = torch.zeros((data_len,feat_dim))


    for i in range(0,data_len,batch_size):
        prompts = test_text_list[i:i+batch_size]
        img_names = test_img_list[i:i+batch_size]
        imgs = []
        for imgname in img_names:
            #print('imgname=',imgname)
            imgs.append(load_image(imgname))

        text_latents = sample_latents(batch_size=batch_size,model=text_model,diffusion=diffusion,
            guidance_scale=text_guidance_scale,model_kwargs=dict(texts=prompts),
            progress=False,clip_denoised=True,use_fp16=True,use_karras=True,karras_steps=64,
            sigma_min=1e-3,sigma_max=160,s_churn=0,)
        img_latents = sample_latents(batch_size=batch_size,model=img_model,diffusion=diffusion,
            guidance_scale=img_guidance_scale,model_kwargs=dict(images=imgs),
            progress=False,clip_denoised=True,use_fp16=True,use_karras=True,karras_steps=64,
            sigma_min=1e-3,sigma_max=160,s_churn=0,)

        #text_features[i:i+batch_size,:]=text_latents.cpu()
        #img_features[i:i+batch_size,:]=img_latents.cpu()
        feature_save_path = os.path.join(text_save_dir,'txt_'+str(i)+'.pt')
        torch.save(text_latents.cpu(), feature_save_path)
        feature_save_path = os.path.join(image_save_dir,'img_'+str(i)+'.pt')
        torch.save(img_latents.cpu(), feature_save_path)
        print('Start index is: ', i)
    
    print('2. Test dataset have been done! It has ',data_len, ' files')
    """    
    '''
    #print('latents.shape=',latents.shape)
    feature_save_path = os.path.join(save_dir,'text_features.pt')
    torch.save(text_features, feature_save_path)

    feature_save_path = os.path.join(save_dir,'image_features.pt')
    torch.save(img_features, feature_save_path)
    '''