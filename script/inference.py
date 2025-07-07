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
    parser.add_argument('--text_file',type=str,default='./datasets/text2shape/test_text.txt', help='directory of test dataset')
    parser.add_argument('--num_threads', type=int, default=8, help='Number of threads for data loading')
    
    # network parameter
    parser.add_argument('--input_dim', type=int, default=768, help='dimensionality of the image and text')
    parser.add_argument('--num_neurons', type=int, default=1024, help='number of neurons per layer')
    parser.add_argument('--num_layers', type=int, default=8, help='number of hidden layers before output')
    parser.add_argument('--full_quadratic', type=bool, default=False, help='if the last layer is full quadratic or not')
    parser.add_argument('--activation', type=str, default='celu', help='which activation to use for')

    # train parameter
    parser.add_argument('--save_dir', type=str, default='./results/test2shape/mesh/', help='Folder for storing obj mesh')#sot-mesh
    parser.add_argument('--text_guidance', type=float, default=15.0, help='size of the batches')
    parser.add_argument('--img_guidance', type=float, default=3.0, help='size of the batches')
    parser.add_argument('--seed', type=int, default=2023, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--OT_model', type=str, default='./training/cvx_0/invx_2.0/lr_1e-05/act_celu/quad_False/layer_8/try_0/ckpt/convex_g_300.pt', help='Folder for storing obj mesh')
    
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

    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    model2 = load_model('image300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    clip_model = FrozenImageCLIP(device)
    convex_f = ICNN_Quadratic(args.num_layers,args.input_dim, args.num_neurons, 
                                args.activation,args.full_quadratic).to(device)
    convex_f.load_state_dict(torch.load(args.OT_model))

    text_guidance_scale = args.text_guidance
    img_guidance_scale = args.img_guidance

    text_cont = []
    with open(args.text_file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n') 
            text_cont.append(line)
   
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir) 

    for cnts in range(len(text_cont)): 
        prompt  = text_cont[cnts] 
        basename = prompt[:240]+'.obj'
                
        text_latents = clip_model(batch_size=1, texts=[prompt])
        text_feat = Variable(text_latents.to(device),requires_grad=True)
        ot_feats = compute_optimal_transport_map(text_feat, convex_f)
        model_kwargs = dict(txt2img_clip=ot_feats)
        latents = generate_latent(model, diffusion, text_guidance_scale, model_kwargs)
        t = decode_latent_mesh(xm, latents[0]).tri_mesh()
        meshName = os.path.join(args.save_dir, basename)
        with open(meshName, 'w') as f:
            t.write_obj(f)