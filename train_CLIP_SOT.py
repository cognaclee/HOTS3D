from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torch.autograd import Variable
from scipy.stats import truncnorm
import numpy as np
import os
import logging

from OT_modules.icnn_modules import ICNN_Quadratic
from OT_modules.all_losses import compute_constraint_loss
from utils.textImage import CreateDataLoader

from shap_e.util.notebooks import decode_latent_mesh
from shap_e.models.download import load_model, load_config
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.diffusion.sample import sample_latents


os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Arguments
def parse_args_and_logger():
    parser = argparse.ArgumentParser(description='OT: text to image')
    #data parameter
    parser.add_argument('--text_dir',type=str,default='/mnt/data2/Objaverse/train/text', help='directory of text features')
    parser.add_argument('--img_dir',type=str,default='/mnt/data2/Objaverse/train/image', help='directory of image features')
    parser.add_argument('--is_train', type=bool, default=True, help='Is it a training phase or not')
    parser.add_argument('--num_threads', type=int, default=8, help='Number of threads for data loading')
    
    # network parameter
    parser.add_argument('--input_dim', type=int, default=768, help='dimensionality of the image and text')#1048576
    parser.add_argument('--num_neurons', type=int, default=1024, help='number of neurons per layer')#16*1024
    parser.add_argument('--num_layers', type=int, default=11, help='number of hidden layers before output')
    parser.add_argument('--mat_rank', type=int, default=32, help='rank of parameter matrix')
    parser.add_argument('--full_quadratic', type=bool, default=True, help='if the last layer is full quadratic or not')
    parser.add_argument('--activation', type=str, default='relu', help='which activation to use for')

    # train parameter
    parser.add_argument('--batch_size', type=int, default=400, help='size of the batches')#720*4999
    parser.add_argument('--total_iters', type=int, default=500, help='number of iterations of training')
    parser.add_argument('--gen_iters', type=int, default=16, help='number of training steps for discriminator per iter')
    
    parser.add_argument('--lambda_cvx', type=float, default=1.0, help='Weight for positive weight constraints')#1e-4
    parser.add_argument('--lambda_fenchel_eq', type=float, default=1e-8, help='Weight for making sure that fenchel equality holds for f,g')#1e-8
    parser.add_argument('--lambda_fenchel_ineq', type=float, default=0.0, help='Weight for making sure that fenchel inequality holds')# non zeros value will result in NAN
    parser.add_argument('--lambda_inverse_x_side', type=float, default=2, help='Weight for making sure that cyclic map')
    
    parser.add_argument('--optimizer', type=str, default='Adam', help='which optimizer to use')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--eps', type=float, default=1e-9)

    # Less frequently used training settings 
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
    parser.add_argument('--seed', type=int, default=2023, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--pretrained_dir', type=str, default='/mnt/data2/lwh/', help='Folder for pretrained models')
    parser.add_argument('--test_dir',type=str,default='/mnt/data2/Objaverse/test', help='directory of test dataset')
    parser.add_argument('--save_dir', type=str, default='./results/OT/', help='Folder for storing obj mesh')
    parser.add_argument('--test_batch_size', type=int, default=10, help='size of the batches for testing')#
    parser.add_argument('--trys', type=int, default=1, help='Orders of try')
    
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()

    args.lr_schedule = 4000

    # ### Seed stuff
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    ####### Storing stuff
    '''
    model_save_path = os.path.join(args.save_dir , 'ckpt')
    sample_save_path = os.path.join(args.save_dir , 'mesh')
    reconstruction_save_path = os.path.join(args.save_dir ,'reconstruction')

    os.makedirs(model_save_path, exist_ok = True)
    os.makedirs(sample_save_path, exist_ok = True)
    os.makedirs(reconstruction_save_path, exist_ok= True)
    '''

    #setup_logging(os.path.join(args.save_dir , 'log.txt'))
    print("run arguments: %s", args)

    return args, logging



## This function computes the optimal transport map given by \nabla convex_g(y)
## Note that 'y' is of size (batch_size, y_dim). Hence the output is also of the same dimension

def compute_optimal_transport_map(X, convex_model, eps=1e-9):

    fx = convex_model.forward(X).sum()
    grad_fx = torch.autograd.grad(fx,X, create_graph=True)[0]
    denom = 1- torch.sum(X*grad_fx,dim=-1,keepdim=True)
    T_X_to_Y = X - grad_fx/(denom+eps)
    #'''
    norm = torch.sqrt(torch.sum(T_X_to_Y*T_X_to_Y,dim=-1,keepdim=True))
    T_X_to_Y = T_X_to_Y/(norm+eps)
    #'''
        
    return T_X_to_Y


def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values



################################################################
def define_network_optimizer(args):
    #print('Have done0030!')
    convex_f = ICNN_Quadratic(args.num_layers,args.input_dim, args.num_neurons, args.activation,args.full_quadratic,args.mat_rank)
    convex_g = ICNN_Quadratic(args.num_layers,args.input_dim, args.num_neurons, args.activation,args.full_quadratic,args.mat_rank)
    ### Form a list of positive weight parameters
    # and also initialize them with positive values
    f_positive_params = []
    for p in list(convex_f.parameters()):
        if hasattr(p, 'be_positive'):
            f_positive_params.append(p)
        p.data = torch.from_numpy(truncated_normal(p.shape, threshold=1./np.sqrt(p.shape[1] if len(p.shape)>1 else p.shape[0]))).float()

    g_positive_params = []
    for p in list(convex_g.parameters()):
        if hasattr(p, 'be_positive'):
            g_positive_params.append(p)
        p.data = torch.from_numpy(truncated_normal(p.shape, threshold=1./np.sqrt(p.shape[1] if len(p.shape)>1 else p.shape[0]))).float()

    if args.cuda:
        convex_f.to(device)
        convex_g.to(device)    

    logging.info("Created and initialized the convex neural networks 'f' and 'g'")
    num_parameters = sum([l.nelement() for l in convex_f.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    if args.optimizer == 'SGD':
        optimizer_f = optim.SGD(convex_f.parameters(), lr=args.lr, momentum=args.momentum)
        optimizer_g = optim.SGD(convex_g.parameters(), lr=args.lr, momentum=args.momentum)
    if args.optimizer == 'Adam':
        optimizer_f = optim.Adam(convex_f.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=1e-5)
        optimizer_g = optim.Adam(convex_g.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=1e-5)

    return convex_f, convex_g, optimizer_f, optimizer_g,f_positive_params,g_positive_params


def train_one_epoch(args,data_loader,convex_f, convex_g, optimizer_f, optimizer_g,f_pos_params,g_pos_params):
    convex_f.train()
    convex_g.train()

    running_w_2_loss = 0.0
    running_g_OT_loss = 0.0
    running_g_cvx_loss = 0.0
    for i, (text_feat,img_feat) in enumerate(data_loader):
        text_feat = text_feat.view((-1,text_feat.shape[-1]))
        img_feat = img_feat.view((-1,img_feat.shape[-1]))
        #print('text_feat.shape=',text_feat.shape)
        #print('img_feat.shape=',img_feat.shape)
        w_2_loss, g_OT_loss, g_cvx_loss = train(args,text_feat,img_feat, convex_f, convex_g, 
            optimizer_f, optimizer_g,f_pos_params,g_pos_params)
        running_w_2_loss += w_2_loss
        running_g_OT_loss += g_OT_loss
        running_g_cvx_loss += g_cvx_loss


    denom = len(data_loader)*args.batch_size
    w_2_loss = running_w_2_loss/denom
    g_OT_loss = running_g_OT_loss/denom
    g_cvx_loss = running_g_cvx_loss/denom

    return w_2_loss, g_OT_loss, g_cvx_loss


def cal_f_loss(args, f_model, g_model, X, Y=None, eps=1e-9):
    fx = f_model.forward(X)
    fx_sum = fx.sum()
    grad_fx = torch.autograd.grad(fx_sum,X, create_graph=True)[0]
    denom = 1- torch.sum(X*grad_fx,dim=-1,keepdim=True)
    T_X_to_Y = X - grad_fx/(denom+eps)
    norm = torch.sqrt(torch.sum(T_X_to_Y*T_X_to_Y,dim=-1,keepdim=True))
    #print('norm.max=',norm.abs().max())
    T_X_to_Y = T_X_to_Y/(norm+eps)
    #print('T_X_to_Y.max=',T_X_to_Y.abs().max())
    #'''
    g_Tx = g_model.forward(T_X_to_Y)

    cost = torch.log(2-torch.sum(X*T_X_to_Y,dim=-1,keepdim=True))
    f_loss = torch.mean(cost - g_Tx)
    loss = f_loss
    #print('f_loss=',f_loss.item())

    #'''
    if args.lambda_inverse_x_side > 0:
        g_grad_Tx = torch.autograd.grad(g_Tx.sum(),T_X_to_Y, create_graph=True)[0]
        denom2 = 1- torch.sum(T_X_to_Y*g_grad_Tx,dim=-1,keepdim=True)
        X2Y2X = T_X_to_Y - g_grad_Tx/(denom2+eps)
        norm = torch.sqrt(torch.sum(X2Y2X*X2Y2X,dim=-1,keepdim=True))
        X2Y2X = X2Y2X/norm
        #constraint_loss = args.lambda_inverse_x_side*(X2Y2X - X).pow(2).sum(dim=-1).mean()
        constraint_loss = args.lambda_inverse_x_side*(X2Y2X*X -1).pow(2).sum(dim=-1).mean()
        loss += constraint_loss
        #print('constraint_loss=',constraint_loss.item())
    #'''

    if args.lambda_fenchel_ineq > 0:
        gy = g_model.forward(Y)
        cost = torch.log(2-torch.sum(X*Y,dim=-1,keepdim=True))
        ineq_reg = args.lambda_fenchel_ineq*(fx+gy-cost).pow(2).mean()
        loss += ineq_reg
        #print('f_ineq_reg=',ineq_reg.item())

    if args.lambda_fenchel_eq > 0:
        cost = torch.log(2-torch.sum(X*T_X_to_Y,dim=-1,keepdim=True))
        eq_reg = args.lambda_fenchel_eq*(fx+g_Tx-cost).pow(2).mean()
        loss += eq_reg
        #print('f_eq_reg=',eq_reg.item())
     
    return loss, f_loss

def cal_g_loss(args, f_model, g_model, X, Y, eps=1e-9):
    fx = f_model.forward(X)
    grad_fx = torch.autograd.grad(fx.sum(),X, create_graph=True)[0]
    denom = 1- torch.sum(X*grad_fx,dim=-1,keepdim=True)
    T_X_to_Y = X - grad_fx/(denom+eps)
    #'''
    norm = torch.sqrt(torch.sum(T_X_to_Y*T_X_to_Y,dim=-1,keepdim=True))
    T_X_to_Y = T_X_to_Y/(norm+eps)
    #'''

    gy = g_model.forward(Y)
    g_loss = torch.mean(g_model.forward(T_X_to_Y) - gy)

    loss = g_loss

    if args.lambda_fenchel_ineq > 0:
        cost = torch.log(2-torch.sum(X*Y,dim=-1,keepdim=True))
        ineq_reg = args.lambda_fenchel_ineq*(fx+gy-cost).pow(2).mean()
        loss += ineq_reg
        #print('g_ineq_reg=',ineq_reg.item())

    cost = torch.log(2-torch.sum(X*T_X_to_Y,dim=-1,keepdim=True))
    if args.lambda_fenchel_eq > 0:
        g_Tx = g_model.forward(T_X_to_Y)
        eq_reg = args.lambda_fenchel_eq*(fx+g_Tx-cost).pow(2).mean()
        loss += eq_reg
        #print('g_eq_reg=',eq_reg.item())
    W2 = (-g_loss+torch.mean(cost)).item()
    return loss,g_loss, W2


def train(args,text_feat,img_feat, convex_f, convex_g, optimizer_f, optimizer_g,f_pos_params,g_pos_params):

    f_OT_loss = 0
    f_cvx_loss = 0

    x = Variable(text_feat.to(device),requires_grad=True)
    y = Variable(img_feat.to(device),requires_grad=True)

    optimizer_f.zero_grad()
    optimizer_g.zero_grad()

    # Train the parameters of 'f'
    for _ in range(1, args.gen_iters+1):
        # First do a forward pass on x and compute grad_f_y
        # Then do a backward pass update on parameters of f

        loss, f_loss = cal_f_loss(args, convex_f, convex_g, x, y)
        f_OT_loss += f_loss.item()

        # So for the last iteration, gradients of 'g' parameters are also updated
        loss.backward()

        ### Constraint loss for g parameters
        if args.lambda_cvx > 0:
            f_positive_loss = args.lambda_cvx*compute_constraint_loss(f_pos_params)
            f_cvx_loss += f_positive_loss.item()
            f_positive_loss.backward() 
            #print('f_positive_loss=',f_positive_loss.item())  

        optimizer_f.step()

        ## Maintaining the positive constraints on the convex_g_params
        if args.lambda_cvx <= 0:
            for p in f_pos_params:
                p.data.copy_(torch.relu(p.data))
        
        optimizer_f.zero_grad()
        optimizer_g.zero_grad()


    f_OT_loss /= args.gen_iters
    if args.lambda_cvx > 0:
        f_cvx_loss /= (args.gen_iters*args.lambda_cvx)

    ## Train the parameters of 'g'
    #x.grad.data.zero_()
    #y.grad.data.zero_()

    loss, g_loss,W2 = cal_g_loss(args, convex_f, convex_g, x, y)

    loss.backward()
    optimizer_g.step()

    # Maintain the "f" parameters positive
    for p in g_pos_params:
        p.data.copy_(torch.relu(p.data))


    return W2, f_OT_loss, f_cvx_loss


###################################################
if __name__ == '__main__':

    args, logging = parse_args_and_logger()
    # Data loader
    train_loader = CreateDataLoader(args)
    test_loader = CreateDataLoader(args,'test')
    # network and optimizer
    pretrained_dir = '/mnt/data2/lwh/'
    convex_f,convex_g,optimizer_f,optimizer_g,f_pos_params,g_pos_params = define_network_optimizer(args)
    #convex_f.load_state_dict(torch.load(pretrained_dir+'/cvx_0.0/eq_0.0/ineq_0.0/invx_1/lr_0.0001/ckpt/convex_f.pt'))
    #convex_g.load_state_dict(torch.load(pretrained_dir+'/cvx_0.0/eq_0.0/ineq_0.0/invx_1/lr_0.0001/ckpt/convex_g.pt'))
    
    decoder = load_model('transmitter', device=device)
    model_clip_t = load_model('text300M', device=device)
    #model_clip_i = load_model('image300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    guidance_scale = 15.0
    
    # args.save_dir = args.save_dir + "cvx_{0}/eq_{1}/ineq_{2}/invx_{3}/lr_{4}".format(args.lambda_cvx,
	# 								args.lambda_fenchel_eq, args.lambda_fenchel_ineq,args.lambda_inverse_x_side,args.lr)
    args.save_dir = args.save_dir + "cvx_{0}/invx_{1}/lr_{2}/act_{3}/quad_{4}/layer_{5}/try_{6}".format(args.lambda_cvx,
									args.lambda_inverse_x_side,args.lr,args.activation,args.full_quadratic,args.num_layers,args.trys)
    model_save_path = os.path.join(args.save_dir , 'ckpt')
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    sample_save_path = os.path.join(args.save_dir , 'mesh')
    if not os.path.exists(sample_save_path):
        os.makedirs(sample_save_path)
    ## Training stuff
    total_w_2_loss_list = []
    for iteration in range(1, args.total_iters + 1):
        w_2_loss, g_OT_loss, g_cvx_loss = train_one_epoch(args,train_loader,convex_f, convex_g, 
                optimizer_f, optimizer_g,f_pos_params,g_pos_params)

        total_w_2_loss_list.append(w_2_loss)
        # total_g_OT_loss_list.append(g_OT_loss)
        # total_g_cvx_loss_list.append(g_cvx_loss)
        if iteration % args.log_interval == 0:
            print('Iteration: {} [{}/{} ({:.0f}%)] g_OT_loss: {:.4f} g_cvx_Loss: {:.4f}  W_2_loss: {:.4f} '.format(
                iteration, iteration, args.total_iters, 100. * iteration / args.total_iters, 
                g_OT_loss, g_cvx_loss, w_2_loss))

        #if True:
        if iteration % 10 == 0:
            '''
            for i, (text_feat,img_feat,text) in enumerate(test_loader):
                text_feat = text_feat.view((-1,text_feat.shape[-1]))
                img_feat = img_feat.view((-1,img_feat.shape[-1]))  
                if i >0:
                    break
            '''
            text_feat,img_feat,text = test_loader.get_random_samples()
            #idx = np.random.randint(0, 400-args.test_batch_size)
            idx = np.random.randint(0, 100-args.test_batch_size)
            text_feat = text_feat[idx:idx+args.test_batch_size,:]
            img_feat = img_feat[idx:idx+args.test_batch_size,:]
            text_feat = Variable(text_feat.to(device),requires_grad=True)
            img_feat = Variable(img_feat.to(device),requires_grad=True)
            #print('text_feat.shape=',text_feat.shape)
            #print('text0.len=',len(text))
            text = text[idx:idx+args.test_batch_size]
            #print('text1.len=',len(text))
            
            transported_y = compute_optimal_transport_map(text_feat, convex_f)
            latents = sample_latents(
                                    batch_size=args.test_batch_size,
                                    model=model_clip_t,
                                    diffusion=diffusion,
                                    guidance_scale=guidance_scale,
                                    model_kwargs=dict(txt2img_clip=transported_y),
                                    progress=True,
                                    clip_denoised=True,
                                    use_fp16=True,
                                    use_karras=True,
                                    karras_steps=64,
                                    sigma_min=1e-3,
                                    sigma_max=160,
                                    s_churn=0,)
            for k, latent in enumerate(latents):
                t = decode_latent_mesh(decoder, latent).tri_mesh()
                #basename = f'text_{iteration}_{k}.obj'
                #print('text[0][k] = ',text[0][k])
                basename = text[k][:200].replace("/", " or ")
                basename = basename + f'_text_{iteration}.obj'
                meshName = os.path.join(sample_save_path, basename)
                with open(meshName, 'w') as f:
                    t.write_obj(f)
            
            # latents = sample_latents(
            #                         batch_size=args.test_batch_size,
            #                         model=model_clip_t,
            #                         diffusion=diffusion,
            #                         guidance_scale=guidance_scale,
            #                         model_kwargs=dict(txt2img_clip=img_feat),
            #                         progress=True,
            #                         clip_denoised=True,
            #                         use_fp16=True,
            #                         use_karras=True,
            #                         karras_steps=64,
            #                         sigma_min=1e-3,
            #                         sigma_max=160,
            #                         s_churn=0,)
            # for k, latent in enumerate(latents):
            #     t = decode_latent_mesh(decoder, latent).tri_mesh()
            #     #basename = f'img_{iteration}_{k}.obj'
            #     basename = text[k][:240]  + f'_img_{iteration}.obj'
            #     meshName = os.path.join(sample_save_path, basename)
            #     with open(meshName, 'w') as f:
            #         t.write_obj(f)
    
        if iteration % args.lr_schedule == 0:
            optimizer_g.param_groups[0]['lr'] = optimizer_g.param_groups[0]['lr'] * 0.5
            optimizer_f.param_groups[0]['lr'] = optimizer_f.param_groups[0]['lr'] * 0.5


        if iteration % 50 == 0:
            filename = os.path.join(model_save_path, 'convex_f_'+str(iteration)+'.pt')
            torch.save(convex_f.state_dict(), filename)
            filename = os.path.join(model_save_path, 'convex_g_'+str(iteration)+'.pt')
            torch.save(convex_g.state_dict(), filename)

    logging.info("Training is finished and the models are saved.")
