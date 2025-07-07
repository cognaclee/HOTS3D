import torch
from torch.utils.data import Dataset
from glob import glob
import numpy as np
import os

class textImageDataset(Dataset):
    '''
    The second class is SphereDataset, which also inherits from the Dataset 
    class in PyTorch. It takes a path to a HDF5 file that contains point clouds 
    and normals of spheres with different radii and positions. 
    It also takes a scale mode to normalize the point clouds, and an optional 
    transform to apply to the data.
    '''
    
    def __init__(self, text_dir, img_dir, text_file=None):
        '''
        This method initializes the class attributes 
        and calls the load_points_normals, get_statistics
        and preprocess methods.
        '''
        super().__init__()
        
        self.text_list = glob(text_dir+'/*.'+'pt')
        self.img_list = glob(img_dir+'/*.'+'pt')
        data_len1 = len(self.text_list)
        data_len2 = len(self.img_list)
        assert data_len1 == data_len2, 'The number of texts and image are not equal'
        self.data_len = data_len1 if data_len1<data_len2 else data_len2
        
        self.text_cont = None
        if text_file is not None:
            self.text_cont = []
            
            with open(text_file, 'r') as f:
                for line in f.readlines():
                    line = line.strip('\n') 
                    self.text_cont.append(line)



    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        text = torch.load(self.text_list[idx])
        img =  torch.load(self.img_list[idx])
        
        if self.text_cont is None:
            return text,img
        else:
            star = idx*400
            cont = self.text_cont[star:star+400]
            return text,img,cont
            
            
    def get_random_samples(self):
        idx = np.random.randint(0, self.data_len)
        texts = torch.load(self.text_list[idx])
        imgs =  torch.load(self.img_list[idx])
        
        ## Previously incorrect version
        #conts = self.text_cont[idx:idx+400]
        
        ## Corrected version
        star = idx*400
        conts = self.text_cont[star:star+400]
        return texts,imgs,conts


def CreateDataLoader(args,phase='train'):
    """loads dataset class"""
    '''
    if opt.dataset_mode == 'meshcnn':
        from data.generation_data import GenerationData
        dataset = GenerationData(opt)
    elif opt.dataset_mode == 'difussionnet':
    '''
    if phase == 'train':
        dataset = textImageDataset(args.text_dir,args.img_dir)
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size//400, shuffle=args.is_train,num_workers=args.num_threads)
        # dataset = TextImageSplitDataset(args.text_dir,args.img_dir,args.batch_size)
        # dataloader = torch.utils.data.DataLoader(dataset,batch_size=1, shuffle=args.is_train,num_workers=args.num_threads)
    else:
        text_dir = os.path.join(args.test_dir , 'text')
        img_dir = os.path.join(args.test_dir , 'image')
        text_file = '/mnt/data2/Objaverse/test/text.txt'
        #dataset = textImageDataset(text_dir, img_dir, text_file)
        #dataloader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size//400, shuffle=False,num_workers=args.num_threads)
        dataloader = textImageDataset(text_dir, img_dir, text_file)
    return dataloader
    
    

class TextImageSplitDataset(Dataset):
    '''
    The second class is SphereDataset, which also inherits from the Dataset 
    class in PyTorch. It takes a path to a HDF5 file that contains point clouds 
    and normals of spheres with different radii and positions. 
    It also takes a scale mode to normalize the point clouds, and an optional 
    transform to apply to the data.
    '''
    
    def __init__(self, text_dir, img_dir, batchsize, text_file=None):
        '''
        This method initializes the class attributes 
        and calls the load_points_normals, get_statistics
        and preprocess methods.
        '''
        super().__init__()
        
        self.text_list = glob(text_dir+'/*.'+'pt')
        self.img_list = glob(img_dir+'/*.'+'pt')
        data_len1 = len(self.text_list)
        data_len2 = len(self.img_list)
        assert data_len1 == data_len2, 'The number of texts and image are not equal'
        self.data_len = data_len1 if data_len1<data_len2 else data_len2
        
        self.batchsize = batchsize
        self.parts = 400//batchsize
        self.data_len = self.data_len*self.parts
        
        
        self.text_cont = None
        if text_file is not None:
            self.text_cont = []
            
            with open(text_file, 'r') as f:
                for line in f.readlines():
                    line = line.strip('\n') 
                    self.text_cont.append(line)



    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        
        b = idx//self.parts
        p = idx%self.parts
        
        st_idx = p*self.batchsize
        text = torch.load(self.text_list[b])[st_idx:st_idx+self.batchsize,:]
        img =  torch.load(self.img_list[b])[st_idx:st_idx+self.batchsize,:]
        
        if self.text_cont is None:
            return text,img
        else:
            st_idx = idx*self.batchsize
            cont = self.text_cont[st_idx:st_idx+self.batchsize]
            return text,img,cont
            
            
    def get_random_samples(self):
        idx = np.random.randint(0, self.data_len)
        
        b = idx//self.parts
        p = idx%self.parts
        
        st_idx = p*self.batchsize
        texts = torch.load(self.text_list[b])[st_idx:st_idx+self.batchsize,:]
        imgs =  torch.load(self.img_list[b])[st_idx:st_idx+self.batchsize,:]
        st_idx = idx*self.batchsize
        conts = self.text_cont[st_idx:st_idx+self.batchsize]
        return texts,imgs,conts
