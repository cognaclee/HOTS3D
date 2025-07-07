import torch
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from shap_e.models.download import load_model
from shap_e.util.data_util import load_or_create_multimodal_batch
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device)
#model_path = "example_data/cactus/object.obj"
#model_path ="/root/data/lwh_code/shap-e/datasets/example_data/cactus/object.obj"
model_path ="/home/lwh/code/shap-e/datasets/example_data/cactus/object.obj"
def save_images(batch,save_dir,basename):
    """ Display a batch of images inline. """
    for i in range(len(batch)):
        #print(type(batch[i]))
        image_pil=batch[i]
        img_name = os.path.join(save_dir, basename+'_'+str(i)+'.png')
        image_pil.save(img_name)

# This may take a few minutes, since it requires rendering the model twice
# in two different modes.
batch = load_or_create_multimodal_batch(
    device,
    model_path=model_path,
    mv_light_mode="basic",
    mv_image_size=256,
    cache_dir="example_data/cactus/cached",
    verbose=True, # this will show Blender output during renders
)
#print('latents.shape=',latents.shape)

save_dir = './results/encode/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with torch.no_grad():
    latent = xm.encoder.encode_to_bottleneck(batch)

    render_mode = 'stf' # you can change this to 'nerf'
    size = 512 # recommended that you lower resolution when using nerf

    cameras = create_pan_cameras(size, device)
    images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
    basename = model_path.split('/')[-1][:-4]
    print(len(images))
    save_images(images,save_dir,basename)