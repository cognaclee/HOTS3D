import torch
from PIL import Image
import os
import time


os.environ["CUDA_VISIBLE_DEVICES"] = '2'
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget


# Example of saving the latents as meshes.
from shap_e.util.notebooks import decode_latent_mesh

idx_GPU = 0
torch.cuda.device_count()
os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % idx_GPU
torch.cuda.set_device(idx_GPU)
torch.cuda.is_available()
torch.cuda.current_device()
device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")


xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))


batch_size = 1
guidance_scale = 15.0

# text_file = '/home/lwh/code/shap-e/text_fun_result.txt'
# text_file = '/mnt/data2/Objaverse/test/text.txt'
# text_file = '/mnt/data2/new.txt' #输入我想生成的文本
text_file = '/home/lwh/code/shap-e/text.txt'

text_cont = []
with open(text_file, 'r') as f:
    for line in f.readlines():
        line = line.strip('\n') 
        text_cont.append(line)

#save_dir = '/home/lwh/code/shap-e/Shape_Objaverse'
# save_dir ="/mnt/data2/results/comparison/shape/"
save_dir ='/home/lwh/code/shap-e/shape_Result'
#save_dir ="/mnt/data2/shape/"  #生成的3D模型的保存目录
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

start_time = time.time()
for cnts in range(len(text_cont)):   
    prompt  = text_cont[cnts] 
    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
        )
    #print(latents.shape)
    t = decode_latent_mesh(xm, latents).tri_mesh()

    basename = prompt[:240].replace("/", " or ")
    basename = str(cnts) + '_' + basename+'.obj'
    meshName = os.path.join(save_dir, basename)

    # basename = prompt[:240]+'.obj'
    # meshName = os.path.join(save_dir, basename)
    with open(meshName, 'w') as f:
        t.write_obj(f)
    # if cnts ==100:                                       这个也是测时间的
    #     end_time = time.time()
    #     ave_time = (end_time - start_time) / cnts
    #     print("end_time is",end_time)
    #     print("ave_time is",ave_time)
    #     break