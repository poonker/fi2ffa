import os.path
import random
import torch
from data.base_dataset import BaseDataset, get_params, get_transform_six_channel
from data.image_folder import make_dataset
from PIL import Image
import re


class fi2ffawboneDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test' also.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # get the image directory
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # get the image directory
        self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')  # get the image directory
        self.dir_D = os.path.join(opt.dataroot, opt.phase + 'D')  # get the image directory               
        self.dir_A_mask = os.path.join(opt.dataroot, opt.phase + 'A_mask')  #
        self.dir_B_mask = os.path.join(opt.dataroot, opt.phase + 'B_mask')  #
        self.dir_C_mask = os.path.join(opt.dataroot, opt.phase + 'C_mask')  #     
        self.dir_D_mask = os.path.join(opt.dataroot, opt.phase + 'D_mask')  #              

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # get image paths
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # get image paths
        self.C_paths = sorted(make_dataset(self.dir_C, opt.max_dataset_size))  # get image paths   
        self.D_paths = sorted(make_dataset(self.dir_D, opt.max_dataset_size))  # get image paths              
        self.A_mask_paths = sorted(make_dataset(self.dir_A_mask, opt.max_dataset_size))  # get image paths
        self.B_mask_paths = sorted(make_dataset(self.dir_B_mask, opt.max_dataset_size))  # get image paths
        self.C_mask_paths = sorted(make_dataset(self.dir_C_mask, opt.max_dataset_size))  # get image paths        
        self.D_mask_paths = sorted(make_dataset(self.dir_D_mask, opt.max_dataset_size))  # get image paths        

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.C_size = len(self.C_paths)
        self.D_size = len(self.D_paths)
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image

        #这里要改
        #self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        #self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        self.isTrain = opt.isTrain

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a degraded image given a random integer index
        A_path = self.A_paths[index % self.A_size]
        A_mask_paths = self.A_mask_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]
        #B_mask_paths = self.B_mask_paths[index % self.B_size]        
        if self.opt.serial_batches:   # make sure index is within then range
            index_C = index % self.C_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_C = random.randint(0, self.C_size - 1)
        C_path = self.C_paths[index_C]
        C_mask_paths = self.C_mask_paths[index_C]  
        D_path = self.D_paths[index_C]
        #D_mask_paths = self.C_mask_paths[index_C]   

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')    
        C_img = Image.open(C_path).convert('RGB')   
        D_img = Image.open(D_path).convert('RGB') 
                  
        A_img_mask = Image.open(A_mask_paths).convert('L')
        #B_img_mask = Image.open(B_mask_paths).convert('L')
        C_img_mask = Image.open(C_mask_paths).convert('L')

        # 对输入和输出进行同样的transform（裁剪也继续采用）

        #20231211修改输入和输出为同一变换
        transform_params = get_params(self.opt, A_img.size)
        A_transform, A_mask_transform = get_transform_six_channel(self.opt, transform_params)
        B_transform, B_mask_transform = get_transform_six_channel(self.opt, transform_params)
        C_transform, C_mask_transform = get_transform_six_channel(self.opt, transform_params)
        D_transform, D_mask_transform = get_transform_six_channel(self.opt, transform_params)
        
        # A_transform_params = get_params(self.opt, A_img.size)
        # A_transform, A_mask_transform = get_transform_six_channel(self.opt, A_transform_params)

        # B_transform_params = get_params(self.opt, B_img.size)
        # B_transform, B_mask_transform = get_transform_six_channel(self.opt, B_transform_params)

        A = A_transform(A_img)
        A_mask = A_mask_transform(A_img_mask)

        B = B_transform(B_img)
        #B_mask = B_mask_transform(B_img_mask)

        C = C_transform(C_img)
        C_mask = C_mask_transform(C_img_mask) 

        D = D_transform(D_img)
        # D_mask = D_mask_transform(D_img_mask)          
        #这里ffabone的mask与ffa的mask一致，不用代入了      
        return {'fi': A, 'fibone': B, 'ffa': C, 'ffabone': D, 
                'fi_mask':A_mask,'fibone_mask':A_mask,'ffa_mask':C_mask,'ffabone_mask':C_mask,
                'fi_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        "degraded images should be in source image folder"
        #return len(self.source_paths)
        return max(self.A_size, self.C_size)