import torch
from PIL import Image
import os
import os
import argparse
from torch.utils.data import Dataset
from torch.autograd import Variable

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

from OT_modules.icnn_modules import ICNN_Quadratic
from utils.textImage import CreateDataLoader
import time
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget

# Example of saving the latents as meshes.
from shap_e.util.notebooks import decode_latent_mesh



# Arguments
def parse_args():
    parser = argparse.ArgumentParser(description='OT: text to image')
    #data parameter
    parser.add_argument('--test_dir',type=str,default='/home/lwh/code/shap-e/text_fun_result.txt', help='directory of test dataset')
    parser.add_argument('--num_threads', type=int, default=8, help='Number of threads for data loading')
    
    # network parameter
    parser.add_argument('--input_dim', type=int, default=768, help='dimensionality of the image and text')
    parser.add_argument('--num_neurons', type=int, default=1024, help='number of neurons per layer')
    parser.add_argument('--num_layers', type=int, default=8, help='number of hidden layers before output')
    parser.add_argument('--full_quadratic', type=bool, default=False, help='if the last layer is full quadratic or not')
    parser.add_argument('--activation', type=str, default='celu', help='which activation to use for')

    # train parameter
    parser.add_argument('--batch_size', type=int, default=50, help='size of the batches')
    parser.add_argument('--save_dir', type=str, default='SOT-fun-result.txt', help='Folder for storing obj mesh')#sot-mesh
    parser.add_argument('--text_guidance', type=float, default=15.0, help='size of the batches')
    parser.add_argument('--img_guidance', type=float, default=3.0, help='size of the batches')
    parser.add_argument('--seed', type=int, default=2023, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--OT_model', type=str, default='./results/OT/cvx_1e-05/invx_2.0/lr_0.0001/act_celu/quad_False/layer_8/try_0/ckpt/convex_g_100.pt', help='Folder for storing obj mesh')
    
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    # ### Seed stuff
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    return args


if __name__ == '__main__':

    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    
    convex_f = ICNN_Quadratic(args.num_layers,args.input_dim, args.num_neurons, 
                                args.activation,args.full_quadratic).to(device)
    convex_f.load_state_dict(torch.load(args.OT_model))

    batch_size = 1
    guidance_scale = 15.0

    text_file = '/home/lwh/code/shap-e/text_fun_result.txt'
    text_cont = []
    with open(text_file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n') 
            text_cont.append(line)
    
    save_dir = '/home/lwh/code/shap-e/test_result.out'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    def compute_optimal_transport_map(y, convex_g):

        g_of_y = convex_g(y).sum()

        grad_g_of_y = torch.autograd.grad(g_of_y, y, create_graph=True)[0]

        return grad_g_of_y

        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))

    convex_f = ICNN_Quadratic(args.num_layers,args.input_dim, args.num_neurons, 
                                    args.activation,args.full_quadratic).to(device)
    convex_f.load_state_dict(torch.load(args.OT_model))

        
    decoder = load_model('transmitter', device=device)
    for cnts in range(len(text_cont)):   
        prompt  = text_cont[cnts] 

        ##########################SOT##############################
        ot_feats = compute_optimal_transport_map(text_feat, convex_f)

        latents = sample_latents(batch_size=batch_size,
                                        model=text_model,
                                        diffusion=diffusion,
                                        guidance_scale=guidance_scale,
                                        model_kwargs=dict(txt2img_clip=ot_feats),
                                        progress=True,
                                        clip_denoised=True,
                                        use_fp16=True,
                                        use_karras=True,
                                        karras_steps=64,
                                        sigma_min=1e-3,
                                        sigma_max=160,
                                        s_churn=0,)
        for i, latent in enumerate(latents):
            t = decode_latent_mesh(xm, latent).tri_mesh()
            basename = prompt[:240]+'.obj'
            meshName = os.path.join(save_dir, basename)
            with open(meshName, 'w') as f:
                t.write_obj(f)