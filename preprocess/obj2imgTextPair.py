import os
import argparse
import numpy as np
import csv
import json
from glob import glob

import torch
from PIL import Image
import blobfile as bf

from shap_e.models.download import load_model
from shap_e.util.data_util import load_or_create_multimodal_batch
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images


CUR_PATH = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--dset', type=str, choices=['shapenet', 'abc', 'pix3d', 'building'], default='shapenet', help='which dataset to extract sdf')
parser.add_argument('--category', type=str, default="all", help='Which single class to generate on [default: all, can '
                                                                'be chair or plane, etc.]')
parser.add_argument('--data_dir', type=str, default="/mnt/data2/text2shape/")
parser.add_argument('--save_root', type=str, default="/mnt/data2/text2shape/dataset/") #./datasets/ShapeNet/PNG/ 这里是转成的image的保存路径
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

seed = 777
np.random.seed(seed)

def split_train_test(dataroot):
    csv_file = f'{dataroot}/text2shape/captions.tablechair.csv'

    assert os.path.exists(csv_file)



    with open(csv_file) as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader, None)
        data = [row for row in reader]

    # split train test
    train_ratio = 0.8
    N = len(data)
    N_train = int(N * train_ratio)

    np.random.shuffle(data)

    train_data = data[:N_train]
    test_data = data[N_train:]


    # sanity check
    train_data_as_str = ['-'.join(d) for d in train_data]
    test_data_as_str = ['-'.join(d) for d in test_data]
    assert len(set(train_data_as_str).intersection(set(test_data_as_str))) == 0

    for phase in ['train', 'test']:
        if phase == 'train':
            data_phase = train_data
        else:
            data_phase = test_data
        out_csv = f'{dataroot}/text2shape/captions.tablechair_{phase}.csv'
        with open(out_csv, 'wt') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(header)
            writer.writerows(data_phase)


def save_images(batch,save_dir,base_name):
    img_path_list = []
    for i in range(len(batch)):
        image_pil=batch[i]
        img_name = os.path.join(save_dir, base_name+str(i)+'.png')
        image_pil.save(img_name)
        img_path_list.append(img_name)
    return img_path_list


def text2ShapeDataset(dataroot, phase='train', cat='all',max_dataset_size=None):
    text_csv = f'{dataroot}/text2shape/captions.tablechair_{phase}.csv'

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
    cats_list = []
    text_list = []

    #for d in tqdm(data, total=len(data), desc=f'readinging text data from {text_csv}'):
    for d in tqdm(data, total=len(data)):
        id, model_id, text, cat_i, synset, subSynsetId = d
            
        if cat_i.lower() not in valid_cats:
            continue
            
        obj_path = f'{dataroot}/ShapeNetCore.v1/{synset}/{model_id}/model.obj'

        if not os.path.exists(obj_path):
            continue
            # {'Chair': 26523, 'Table': 33765} vs {'Chair': 26471, 'Table': 33517}
            # not sure why there are some missing files
                
        model_list = model_list.append(obj_path)
        text_list = text_list.append(text)
        cats_list = cats_list.append(synset)


    if max_dataset_size is not None:
        model_list = model_list[:max_dataset_size]
        text_list = text_list[:max_dataset_size]
        cats_list = cats_list[:max_dataset_size]
    print('[*] %d samples loaded.' % (len(model_list)))

    return model_list, text_list, cats_list



def obj_to_images_text_pair(args,model_list,text_list,render_model,cameras):
    start_model = FLAGS.start_model
    end_model = FLAGS.end_model
    if end_model<=start_model:
        end_model = len(model_list)

    all_img_path_list = []
    all_text_list = []

    for k in range(start_model,end_model):
        mesh_file = model_list[k]
        text = text_list[k]
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
            #start_time = time.time()
            latent = render_model.encoder.encode_to_bottleneck(batch)
            images = decode_latent_images(render_model, latent, cameras, rendering_mode=render_mode)
            baseName = text_list[k]
            one_img_path_list = save_images(images,save_dir,baseName)

        all_img_path_list = all_img_path_list+one_img_path_list
        all_text_list = all_text_list + [text]*len(one_img_path_list)

    ###############################################         
    name_file = os.path.join(save_dir,'name.txt')
    with open(name_file, 'w') as fp:
        for name in all_img_path_list:
            fp.write(name+'\r\n')

    text_file = os.path.join(save_dir,'text.txt')
    with open(text_file, 'w') as fp:
        for text in all_text_list:
            fp.write(text+'\r\n') 



if __name__ == "__main__":

    split_train_test(FLAGS.data_dir)

    render_model = load_model('transmitter', device=device)
    render_mode = 'stf' # you can change this to 'nerf'
    size = 512 # recommended that you lower resolution when using nerf
    cameras = create_pan_cameras(size, device)

    # This may take a few minutes, since it requires rendering the model twice
    # in two different modes.
    SAVE_ROOT = FLAGS.save_root
    model_list, text_list, _ = text2ShapeDataset(FLAGS.data_dir, phase='train', cat='all')
    FLAGS.save_root = os.path.join(SAVE_ROOT,'train')
    obj_to_images_text_pair(FLAGS,model_list,text_list,render_model,cameras)

    model_list, text_list, _ = text2ShapeDataset(FLAGS.data_dir, phase='test', cat='all')
    FLAGS.save_root = os.path.join(SAVE_ROOT,'test')
    obj_to_images_text_pair(FLAGS,model_list,text_list,render_model,cameras)


