import torch
from PIL import Image
import os

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.image_util import load_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device)
model = load_model('image300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))
batch_size = 4
guidance_scale = 3.0

# To get the best result, you should remove the background and show only the object of interest to the model.
image = load_image("./datasets/example_data/corgi.png")

latents = sample_latents(batch_size=batch_size,
    model=model,
    diffusion=diffusion,
    guidance_scale=guidance_scale,
    model_kwargs=dict(images=[image] * batch_size),
    progress=True,
    clip_denoised=True,
    use_fp16=True,
    use_karras=True,
    karras_steps=64,
    sigma_min=1e-3,
    sigma_max=160,
    s_churn=0,
)
render_mode = 'nerf' # you can change this to 'stf' for mesh rendering
size = 64 # this is the size of the renders; higher values take longer to render.

print('latents.shape=',latents.shape)

def save_images(batch,save_dir,basename):
    """ Display a batch of images inline. """
    for i in range(len(batch)):
        #print(type(batch[i]))
        image_pil=batch[i]
        img_name = os.path.join(save_dir, basename+'_'+str(i)+'.png')
        image_pil.save(img_name)
    
save_dir = './results/img2mesh/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
'''    
cameras = create_pan_cameras(size, device)
for i, latent in enumerate(latents):
    images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
'''
    
# Example of saving the latents as meshes.
from shap_e.util.notebooks import decode_latent_mesh
for i, latent in enumerate(latents):
    t = decode_latent_mesh(xm, latent).tri_mesh()
    basename = f'example_mesh_{i}.obj'
    meshName = os.path.join(save_dir, basename)
    with open(meshName, 'w') as f:
        t.write_obj(f)
