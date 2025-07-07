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
parser.add_argument('--dset', type=str, choices=['shapenet', 'abc', 'pix3d', 'building'], default='shapenet', help='which dataset to extract sdf')
parser.add_argument('--category', type=str, default="all", help='Which single class to generate on [default: all, can '
                                                                'be chair or plane, etc.]')
#parser.add_argument('--data_dir', type=str, default="/home/lwh/code/shap-e/ab_OBJ/cv0_invx0_OBJ") #'/data/result/OT/mesh/'  这里是TAPS3D的输出的obj
parser.add_argument('--data_dir', type=str, default="/mnt/data2/results/hunyuan3d/Z-calculate/")
parser.add_argument('--save_root', type=str, default="/mnt/data2/results/hunyuan3d/Zhunyuan2png/") #./datasets/ShapeNet/PNG/ 这里是转成的image的保存路径
parser.add_argument('--num_trial', type=int, default=1, help='Index of trial')
parser.add_argument('--GPU', type=int, default=5, help='Index of GPU')
parser.add_argument('--start_model', type=int, default=0, help='Index of start_model')#4780
parser.add_argument('--end_model', type=int, default=0, help='Index of end_model')#6578

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

    text_file ='/mnt/data2/Objaverse/test/text.txt'  #test.txt是用于生成obj文件的那个text
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
    #start_model = 682
    start_model = FLAGS.start_model
    end_model = FLAGS.end_model
    if end_model<=start_model:
        end_model = len(text_cont)
    print('start_model=',start_model)
    print('end_model=',end_model)
    print('num_trial=',FLAGS.num_trial)
    print('GPU=',FLAGS.GPU)
  
    for k in range(0,len(text_cont)):
        #mesh_file = list_obj[k]
        print(text_cont[k])
        #basename = text_cont[k][:240] + '.obj'
        #basename = f"{k}_{text_cont[k].strip()}_0.obj"[:240]
        #basename = text_cont[k][:240] + '_text_120.obj'
        
        mesh_file = os.path.join(FLAGS.data_dir,basename)
        
        if not os.path.exists(mesh_file):
            print('no',mesh_file)
            continue
        
        save_dir = os.path.join(FLAGS.save_root,str(k))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        '''    
        if k%2==1:
            baseName = bf.basename(mesh_file)[:240]
            save_images(images,save_dir,baseName)
            continue
        '''
        
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

        with torch.no_grad():
            start_time = time.time()
            latent = xm.encoder.encode_to_bottleneck(batch)
            images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
            #print(mesh_file)
            #print(len(images))
            # save_images(images,save_dir)
            #save_images(images,save_dir,text_cont[k][:240])
            baseName = bf.basename(mesh_file)[:240]
            save_images(images,save_dir,baseName)
            end_time = time.time()
            run_time=end_time - start_time
            print(f"!!! Run time is: {run_time} s")
    #'''

"""

import os
import argparse
import numpy as np
import json
import time
from tqdm import tqdm
from glob import glob
from concurrent.futures import ProcessPoolExecutor

import torch
from PIL import Image
import blobfile as bf

from shap_e.models.download import load_model
from shap_e.util.data_util import load_or_create_multimodal_batch
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images

CUR_PATH = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--dset', type=str, choices=['shapenet', 'abc', 'pix3d', 'building'], default='shapenet', help='which dataset to extract sdf')
parser.add_argument('--category', type=str, default="all", help='Which single class to generate on [default: all, can be chair or plane, etc.]')
parser.add_argument('--data_dir', type=str, default="/mnt/data2/results/hunyuan3d/Z-calculate/")
parser.add_argument('--save_root', type=str, default="/mnt/data2/results/hunyuan3d/Zhunyuan2png/")
parser.add_argument('--num_trial', type=int, default=1, help='Index of trial')
parser.add_argument('--GPU', type=int, default=5, help='Index of GPU')
parser.add_argument('--start_model', type=int, default=0, help='Index of start_model')
parser.add_argument('--end_model', type=int, default=0, help='Index of end_model')

FLAGS = parser.parse_args()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#保存图像，遍历输入的图像，然后将其按一定的命名规则保存
def save_images(batch, save_dir, base_name):
    
    for i, image_pil in enumerate(batch):
        img_name = os.path.join(save_dir, f"{base_name}{i}.png")
        image_pil.save(img_name)

#处理单个网格文件
def process_mesh_file(mesh_file, k, save_root):
    
    #basename = f"{k}_{bf.basename(mesh_file)[:240]}_0.obj"
    #basename = f"{k + 1}_{text_cont[k].strip()}.0.obj" 
    #基于一个传入的索引k，创建一个保存目录
    save_dir = os.path.join(save_root, str(k))
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cache_dir = "./datasets/cached" + str(FLAGS.num_trial)
    batch = load_or_create_multimodal_batch(
        device,
        model_path=mesh_file,
        mv_light_mode="basic",
        mv_image_size=256,
        cache_dir=cache_dir,
        verbose=True)

    cachename = bf.basename(mesh_file)[:100]
    file_name = bf.join(cache_dir, f"mv_{cachename}_basic_20.zip")
    os.remove(file_name)

    with torch.no_grad():
        start_time = time.time()
        latent = xm.encoder.encode_to_bottleneck(batch)
        images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
        save_images(images, save_dir, basename)
        end_time = time.time()
        run_time = end_time - start_time
        print(f"Processed {mesh_file}: Run time is {run_time:.2f} s")

#加载包含网格文件名称的文本文件，加载路径并返回列表
def load_text_file(text_file):
    
    with open(text_file, 'r') as f:
        return [line.strip() for line in f.readlines()]

if __name__ == "__main__":
    text_file = '/mnt/data2/Objaverse/test/text.txt'
    #应该是改这个text_cont，给text_cont加上序号，并返回_前的字符
    text_cont = load_text_file(text_file)
    #给text_cont前边加上xx_,然后取_之前的
    
    #遍历 text_cont 并为每个元素加上索引
    text_cont = [f"{i}_{line}" for i, line in enumerate(text_cont)]

    print('Loaded mesh files:', len(text_cont))

    xm = load_model('transmitter', device=device)
    render_mode = 'stf'
    size = 512
    cameras = create_pan_cameras(size, device)

    start_model = FLAGS.start_model
    end_model = FLAGS.end_model if FLAGS.end_model > start_model else len(text_cont)

    print('Processing models from', start_model, 'to', end_model)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for k in tqdm(range(start_model, end_model), desc="Processing mesh files"):
            mesh_file = os.path.join(FLAGS.data_dir, f"{text_cont[k].strip()}_0.obj")
            if os.path.exists(mesh_file):
                futures.append(executor.submit(process_mesh_file, mesh_file, k, FLAGS.save_root))
            else:
                print('Mesh file does not exist:', mesh_file)

        # Wait for all tasks to complete
        for future in tqdm(futures,desc="Waiting for processing",total=len(futures)):
            future.result()  # Get result to handle exceptions if any
"""


