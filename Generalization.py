import torch
from PIL import Image
import blobfile as bf
import os
import shutil
import argparse
import numpy as np
import json
from glob import glob
import csv
from tqdm import tqdm
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
import time
import random
from torch.autograd import Variable
# Example of saving the latents as meshes.
from shap_e.util.notebooks import decode_latent_mesh
from OT_modules.icnn_modules import ICNN_Quadratic
from shap_e.models.generation.pretrained_clip import FrozenImageCLIP


idx_GPU = 0
torch.cuda.device_count()
os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % idx_GPU
torch.cuda.set_device(idx_GPU)
torch.cuda.is_available()
torch.cuda.current_device()
device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")


# Arguments
def parse_args():
    parser = argparse.ArgumentParser(description='OT: text to image')
    #data parameter
    parser.add_argument('--test_dir',type=str,default='./datasets/ShapeNet/feature/test', help='directory of test dataset')
    parser.add_argument('--num_threads', type=int, default=8, help='Number of threads for data loading')
    
    # network parameter
    parser.add_argument('--input_dim', type=int, default=768, help='dimensionality of the image and text')
    parser.add_argument('--num_neurons', type=int, default=1024, help='number of neurons per layer')
    parser.add_argument('--num_layers', type=int, default=8, help='number of hidden layers before output')
    parser.add_argument('--full_quadratic', type=bool, default=False, help='if the last layer is full quadratic or not')
    parser.add_argument('--activation', type=str, default='celu', help='which activation to use for')

    # train parameter
    parser.add_argument('--save_dir', type=str, default='/home/lwh/code/shap-e/SOT-fun-result', help='Folder for storing obj mesh')#sot-mesh
    parser.add_argument('--text_guidance', type=float, default=15.0, help='size of the batches')
    parser.add_argument('--img_guidance', type=float, default=3.0, help='size of the batches')
    parser.add_argument('--seed', type=int, default=2023, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--OT_model', type=str, default='./results/OT/cvx_0/invx_2.0/lr_1e-05/act_celu/quad_False/layer_8/try_0/ckpt/convex_g_300.pt', help='Folder for storing obj mesh')
    # parser.add_argument('--OT_model', type=str, default='./results/OT/cvx_1e-05/invx_2.0/lr_0.0001/act_celu/quad_False/layer_8/try_0/ckpt/convex_g_100.pt', help='Folder for storing obj mesh')
    
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    # ### Seed stuff
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    return args

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
        
        #print('png_path=', png_path)
        #print('model_name=', model_name)

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



def compute_optimal_transport_map(y, convex_g):

    g_of_y = convex_g(y).sum()

    grad_g_of_y = torch.autograd.grad(g_of_y, y, create_graph=True)[0]

    return grad_g_of_y
    
def generate_latent(model, diffusion, guidance_scale, kargs):
    
    latents = sample_latents(
            batch_size=1,
            model=model,
            diffusion=diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=kargs,
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
            )
    return latents

###################################################
if __name__ == '__main__':

    args = parse_args()
    
    dataroot='/home/lwh/code/shap-e/datasets'
    batch_size = 400
    '''
    phase = 'test'#'test'
    max_dataset_size = 3600
    img_list, text_list, name_list = text2ShapeDataset(dataroot, 
                                            phase=phase, cat='all',max_dataset_size=max_dataset_size)
    '''

    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    model2 = load_model('image300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    clip_model = FrozenImageCLIP(device)
    convex_f = ICNN_Quadratic(args.num_layers,args.input_dim, args.num_neurons, 
                                args.activation,args.full_quadratic).to(device)
    convex_f.load_state_dict(torch.load(args.OT_model))

    batch_size = 1
    text_guidance_scale = args.text_guidance
    img_guidance_scale = args.img_guidance

    
    text_file = '/home/lwh/code/shap-e/text_fun_result.txt'
    text_cont = []
    with open(text_file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n') 
            text_cont.append(line)
   
    save_dir = '/home/lwh/code/shap-e/SOT-fun-result'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    '''   
    print('[*] %d sample pairs loaded.' % (len(img_list)))
    print('[*] %d texts loaded.' % (len(text_list)))
    save_text_file = os.path.join('/data/result/OT/','text2.txt')
    with open(save_text_file, 'w') as fp:
        for text in text_list:
            new_text = text.replace("\n", " ")
            fp.write(new_text+'\r\n')
    
    save_name_file = os.path.join('/data/result/OT/','name2.txt')
    with open(save_name_file, 'w') as fp:
        for name in img_list:
            fp.write(name+'\r\n')    
      
    save_name_file = '/home/lwh/code/shap-e/datasets/ShapeNet/feature/test/obj.txt'
    with open(save_name_file, 'w') as fp:
        for name in name_list:
            fp.write(name+'\r\n')    
    '''    
    obj_dir = '/data/3D/ShapeNetCore.v1'
    for cnts in range(len(text_cont)): 
        prompt  = text_cont[cnts] 
        basename = prompt[:240]+'.obj'
                
        text_latents = clip_model(batch_size=1, texts=[prompt])
        text_feat = Variable(text_latents.to(device),requires_grad=True)
        ot_feats = compute_optimal_transport_map(text_feat, convex_f)
        model_kwargs = dict(txt2img_clip=ot_feats)
        latents = generate_latent(model, diffusion, text_guidance_scale, model_kwargs)
        print(type(latents))
        print('latents.shape=',latents.shape)
        t = decode_latent_mesh(xm, latents[0]).tri_mesh()
        meshName = os.path.join(save_dir, basename)
        with open(meshName, 'w') as f:
            t.write_obj(f)