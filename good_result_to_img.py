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


CUR_PATH = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--dset', type=str, choices=['shapenet', 'abc', 'pix3d', 'building'], default='shapenet', help='which dataset to extract sdf')
parser.add_argument('--category', type=str, default="all", help='Which single class to generate on [default: all, can '
                                                                'be chair or plane, etc.]')
parser.add_argument('--data_dir', type=str, default="/data/result/our_good_reslut3/slected") #'/data/result/OT/mesh/'
parser.add_argument('--save_root', type=str, default="/data/result/our_good_reslut3/slected-PNG") #./datasets/ShapeNet/PNG/
parser.add_argument('--num_trial', type=int, default=1, help='Index of trial')
parser.add_argument('--GPU', type=int, default=2, help='Index of GPU')
parser.add_argument('--start_model', type=int, default=0, help='Index of start_model')#4780
parser.add_argument('--end_model', type=int, default=0, help='Index of end_model')#6578

FLAGS = parser.parse_args()


os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_images(batch,save_dir,base_name):
    """ Display a batch of images inline. """
    for i in range(len(batch)):
        #print(type(batch[i]))
        image_pil=batch[i]
        print('image_pil=',image_pil.size)
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
    '''
    cat = FLAGS.category

    info_file = './datasets/dataset_info_files/info-shapenet.json'
    with open(info_file) as json_file:
        info_data = json.load(json_file)
        lst_dir, cats, raw_dirs = info_data["lst_dir"], info_data['cats'], info_data['raw_dirs_v1']

    if cat != 'all':
        cats = {cat: cats[cat]}

    start_idx = len(raw_dirs["mesh_dir"])
    list_obj = load_shapenet(lst_dir, cats, raw_dirs["mesh_dir"])
    
    '''


    obj_file = glob(FLAGS.data_dir+'/*.'+'obj')
    
    #'''
    xm = load_model('transmitter', device=device)
    render_mode = 'stf' # you can change this to 'nerf'/stf
    size = 512 # recommended that you lower resolution when using nerf
    cameras = create_pan_cameras(size, device)

  
    for k in range(len(obj_file)):
        #mesh_file = list_obj[k]
        #basename = text_cont[k][:240] + '.obj'
        #basename = text_cont[k][:240] + '_text_120.obj'
        #mesh_file = os.path.join(FLAGS.data_dir,basename)
        mesh_file = obj_file[k]
        if not os.path.exists(mesh_file):
            print('no',mesh_file)
            continue
        cache_dir = "./datasets/cached"+str(FLAGS.num_trial)
        batch = load_or_create_multimodal_batch(
                device,
                model_path=mesh_file,
                mv_light_mode="basic",
                mv_image_size=256,
                cache_dir=cache_dir,
                verbose=True)
        #file_name = cache_dir + "/mv_model.obj_basic_20.zip"
        #print('mesh_file=',mesh_file)
        #print('bf.basename(mesh_file)=',bf.basename(mesh_file))
        cachename = bf.basename(mesh_file)[:100]
        print('cachename=',cachename)
        file_name = bf.join(cache_dir, f"mv_{cachename}_basic_20.zip")
        os.remove(file_name)
        

        save_dir = os.path.join(FLAGS.save_root,str(k))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


        with torch.no_grad():
            latent = xm.encoder.encode_to_bottleneck(batch)
            images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
            #print(mesh_file)
            #print(len(images))
            # save_images(images,save_dir)
            #save_images(images,save_dir,text_cont[k][:240])
            baseName = bf.basename(mesh_file)
            save_images(images,save_dir,baseName)
    #'''