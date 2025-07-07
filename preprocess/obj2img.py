import os
import argparse
import numpy as np
import json
from glob import glob

import torch
from PIL import Image
import blobfile as bf

from shap_e.models.download import load_model
from shap_e.util.data_util import load_or_create_multimodal_batch
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget

import os
import time

CUR_PATH = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="./results/test2shape/mesh")
parser.add_argument('--save_root', type=str, default="./results/text2shape/png") 
parser.add_argument('--num_trial', type=int, default=1, help='Index of trial')
parser.add_argument('--GPU', type=int, default=5, help='Index of GPU')

FLAGS = parser.parse_args()


os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Display a batch of images inline.
def save_images(batch,save_dir,base_name):
    
    for i in range(len(batch)):
        image_pil=batch[i]
        img_name = os.path.join(save_dir, base_name+str(i)+'.png')
        image_pil.save(img_name)


def load_shapenet(lst_dir, cats, mesh_dir):

    # sanity check: all files exists
    for catnm in cats.keys():

        print(f'[*] checking obj files in {catnm} ({cats[catnm]})')

        cat_id = cats[catnm]
        cat_mesh_dir = os.path.join(mesh_dir, cat_id)
        with open(lst_dir+"/"+str(cat_id)+"_test.lst", "r") as f:
            list_obj = f.readlines()

        with open(lst_dir+"/"+str(cat_id)+"_train.lst", "r") as f:
            list_obj += f.readlines()

        list_obj = [f.rstrip() for f in list_obj]
        list_obj = [f'{cat_mesh_dir}/{f}/model.obj' for f in list_obj]

        for f in list_obj:
            if not os.path.exists(f):
                print(f)
                import pdb; pdb.set_trace()
            assert os.path.exists(f)


        print(f'[*] all files exist for {catnm} ({cats[catnm]})!')
    return list_obj




if __name__ == "__main__":

    text_file ='./datasets/text2shape/test_text.txt' 
    text_cont = []
    with open(text_file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n') 
            text_cont.append(line)
    print('text_cont=',len(text_cont))

    #text_cont = glob(FLAGS.data_dir+'/*.'+'obj')
    
    #'''
    xm = load_model('transmitter', device=device)
    render_mode = 'stf' # you can change this to 'nerf'
    size = 512 # recommended that you lower resolution when using nerf
    cameras = create_pan_cameras(size, device)

    # This may take a few minutes, since it requires rendering the model twice
    # in two different modes.
  
    for k in range(0,len(text_cont)):
        #mesh_file = list_obj[k]
        print(text_cont[k])
        basename = text_cont[k][:240] + '.obj'
        #basename = f"{k}_{text_cont[k].strip()}_0.obj"[:240]
        #basename = text_cont[k][:240] + '_text_120.obj'
        
        mesh_file = os.path.join(FLAGS.data_dir,basename)
        
        if not os.path.exists(mesh_file):
            print('no',mesh_file)
            continue
        
        save_dir = os.path.join(FLAGS.save_root,str(k))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        cache_dir = "./datasets/cached"+str(FLAGS.num_trial)
        batch = load_or_create_multimodal_batch(
                device,
                model_path=mesh_file,
                mv_light_mode="basic",
                mv_image_size=256,
                cache_dir=cache_dir,
                verbose=True)
        cachename = bf.basename(mesh_file)[:100]
        print('cachename=',cachename)
        file_name = bf.join(cache_dir, f"mv_{cachename}_basic_20.zip")
        os.remove(file_name)

        with torch.no_grad():
            start_time = time.time()
            latent = xm.encoder.encode_to_bottleneck(batch)
            images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
            baseName = bf.basename(mesh_file)[:240]
            save_images(images,save_dir,baseName)
            end_time = time.time()
            run_time=end_time - start_time
            #print(f"!!! Run time is: {run_time} s")