# -*- coding: UTF-8 -*-
#使用gfenet作为生成器主体

"""
@Function:
@File: fi-ffa-wbone_cycle_model.py
@Date: 2024/7/8 17:04 
@Author: funky
"""

import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from models.guided_filter_pytorch.HFC_filter import HFCFilter

class Fi2Ffawb2fiModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        #poolsize是否保留需要检查显存占用
        #parser.set_defaults(pool_size=0, gan_mode='vanilla')
        if is_train:
            #oneside
            parser.add_argument('--lambda_A', type=float, default=1.0, help='weight for cycle loss (A -> B -> A)')
            #parser.add_argument('--lambda_A_idt', type=float, default=5.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_A_D_fibone', type=float, default=1.0, help='weight for DHF loss')
            parser.add_argument('--lambda_A_D_ffa', type=float, default=1.0, help='weight for DIM loss')  
            parser.add_argument('--lambda_A_DG_fibone', type=float, default=1.0, help='weight for DHF loss')
            parser.add_argument('--lambda_A_DG_ffa', type=float, default=1.0, help='weight for DIM loss')             
            parser.add_argument('--lambda_A_G_fibone1', type=float, default=1.0, help='weight for G gan loss')
            parser.add_argument('--lambda_A_G_ffa', type=float, default=1.0, help='weight for G gan loss')             
            parser.add_argument('--lambda_A_G_fi', type=float, default=1, help='weight for G gan loss')
            parser.add_argument('--lambda_A_G_fibone2', type=float, default=1.0, help='weight for G gan loss')

            #otherside:
            parser.add_argument('--lambda_B', type=float, default=1.0, help='weight for cycle loss (A -> B -> A)')
            #parser.add_argument('--lambda_B_idt', type=float, default=5.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B_D_fi', type=float, default=1.0, help='weight for DHF loss')
            parser.add_argument('--lambda_B_D_ffabone', type=float, default=1.0, help='weight for DIM loss')  
            parser.add_argument('--lambda_B_DG_fi', type=float, default=1.0, help='weight for DHF loss')
            parser.add_argument('--lambda_B_DG_ffabone', type=float, default=1.0, help='weight for DIM loss')              
            parser.add_argument('--lambda_B_G_fi', type=float, default=1.0, help='weight for G gan loss')
            parser.add_argument('--lambda_B_G_ffabone1', type=float, default=1.0, help='weight for G gan loss') 
            parser.add_argument('--lambda_B_G_ffa', type=float, default=1, help='weight for G gan loss')
            parser.add_argument('--lambda_B_G_ffabone2', type=float, default=1.0, help='weight for G gan loss')

            #parser.add_argument('--netG_A', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
            #parser.add_argument('--netD_A_H', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
            # parser.add_argument('--netG_fi2ffabone', type=str, default='unet_gfe_net', help='weight for L1L loss')
            # parser.add_argument('--netG_ffabone2ffa', type=str, default='unet_gfe_net', help='weight for L1L loss')
            # parser.add_argument('--netD_fihf', type=str, default='basic', help='weight for L1L loss')
            # parser.add_argument('--netD_ffabone', type=str, default='basic', help='weight for L1L loss')
            # parser.add_argument('--netD_fi', type=str, default='basic', help='weight for L1L loss')
            # parser.add_argument('--netD_ffa', type=str, default='basic', help='weight for L1L loss')

            #other:
            #parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--RMS', type=str, default='True')
            #parser.add_argument('--PTWH', type=float, default=0.8, help='probability train with high frequency')
            #parser.add_argument('--TWHF', type=float, default=0.0, help='train weight loss with high frequency')
            # parser.add_argument('--outline', type=float, default=10.0, help='train weight loss with high frequency')
            # parser.add_argument('--inline', type=float, default=10.0, help='train weight loss with high frequency')
            # parser.add_argument('--crossline', type=float, default=10.0, help='train weight loss with high frequency')
            #parser.add_argument('--g_im', type=float, default=10.0, help='train weight loss with high frequency')
            #parser.add_argument('--adv_im', type=float, default=1.0, help='train weight loss with high frequency')
            #parser.add_argument('--g_hf', type=float, default=5.0, help='train weight loss with high frequency')
            #parser.add_argument('--adv_hf', type=float, default=1.0, help='train weight loss with high frequency')                        
        parser.add_argument('--filter_width', type=int, default=53, help='weight for G loss')
        parser.add_argument('--nsig', type=int, default=9, help='weight for G loss')
        parser.add_argument('--sub_low_ratio', type=float, default=1.0, help='weight for L1L loss')



        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        #self.input_nc = opt.input_nc
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['D_A_IC','D_A_IM','D_A_IC_HF','D_A_IC_IC',
        #                    'G_A_G','G_A_D','G_A_IC',
        #                    'G_A_G_IM','G_A_G_IC','G_A_G_IC_IMIC','G_A_G_IC_HFIC','G_A_G_IC_IMHF',
        #                    'G_A_D_IM', 'G_A_D_IC',
        #                    'D_B_IC','D_B_IM','D_B_IC_HF','D_B_IC_IC',
        #                    'G_B_G','G_B_D','G_B_IC',
        #                    'G_B_G_IM','G_B_G_IC','G_B_G_IC_IMIC','G_B_G_IC_HFIC','G_B_G_IC_IMHF',
        #                    'G_B_D_IM', 'G_B_D_IC',
        #                    ]
        # self.loss_names = ['G_A_ffabone','G_A_fihf','G_A_fi','G_A_ffa',
        #                    'G_B_fi','G_B_ffa','G_B_ffabone','G_B_fihf',
        #                    'G_adv',
        #                    'G_A','G_B','G',
        #                    'D_fihf','D_ffabone',
        #                    'D_fi','D_ffa',
        #                 #    'D'
        #                     ]
        self.loss_names = ['G_A_ffa','G_A_fibone1','G_A_fi','G_A_fibone2',
                           'G_adv',
                           'G_A','G',
                           'D_fibone','D_ffa',
                        #    'D'
                            ]        
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_fi', 'fake_fibone','fake_ffa', 'rec_fi','rec_fibone','real_fibone']
        #visual_names_B = ['real_ffa', 'fake_fi', 'fake_ffabone','rec_ffa','rec_ffabone','real_ffabone']
        # if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
        #     visual_names_A.append('idt_B')
        #     visual_names_B.append('idt_A')
        #self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        self.visual_names = visual_names_A 
        # guide filter
        self.hfc_filter = HFCFilter(opt.filter_width, nsig=opt.nsig, sub_low_ratio=opt.sub_low_ratio, sub_mask=True, is_clamp=True).to(self.device)

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_fi2ffa', 'G_ffa2fi', 'D_fibone','D_ffa']
        else:  # during test time, only load Gs
            self.model_names = ['G_fi2ffa', 'G_ffa2fi']
            #这一句指定了test的输出内容
            #未必
            self.visual_names = ['rec_ffa']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        #这部分的opt.input_nc, opt.output_nc在生成器不同时需要调整
        self.netG_fi2ffa = networks.define_G(opt.G_A_input_nc, opt.output_nc, opt.ngf, opt.netG_fi2ffa, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_ffa2fi = networks.define_G(opt.G_B_input_nc, opt.output_nc, opt.ngf, opt.netG_ffa2fi, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        #同理
        if self.isTrain:  # define discriminators
            self.netD_fibone = networks.define_D(opt.D_A_input_nc, opt.ndf, opt.netD_fibone,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_ffa = networks.define_D(opt.D_A_input_nc, opt.ndf, opt.netD_ffa,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)            
            #self.netD_fi = networks.define_D(opt.D_B_input_nc, opt.ndf, opt.netD_fi,
                                            #opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            #self.netD_ffabone = networks.define_D(opt.D_B_input_nc, opt.ndf, opt.netD_ffabone,
                                            #opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            # #需要确定
            # if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
            #     assert(opt.input_nc == opt.output_nc)
            # #需要确定
            # self.fake_A_IC_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # self.fake_B_IC_pool = ImagePool(opt.pool_size)
            # self.fake_A_IM_pool = ImagePool(opt.pool_size)
            # self.fake_B_IM_pool = ImagePool(opt.pool_size)
            # self.fake_A_IM_HF_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # self.fake_B_IM_HF_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            # self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            # optimizers
            # 增加使用RMS
            if not self.opt.RMS:
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_fi2ffa.parameters(), self.netG_ffa2fi.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_fibone.parameters(),self.netD_ffa.parameters(),
                                                                    
                                                                    ), lr=opt.lr,
                                                    betas=(opt.beta1, 0.999))
            else:
                self.optimizer_G = torch.optim.RMSprop(itertools.chain(self.netG_fi2ffa.parameters(), self.netG_ffa2fi.parameters()), lr=opt.lr, alpha=0.9)
                self.optimizer_D = torch.optim.RMSprop(itertools.chain(self.netD_fibone.parameters(),self.netD_ffa.parameters(),
                                                                    
                                                                    ), lr=opt.lr,
                                                    alpha=0.9)            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        #AtoB = self.opt.direction == 'AtoB'
        self.real_fi = input['fi'].to(self.device)
        self.real_fibone = input['fibone'].to(self.device)
        #self.real_ffabone = input['ffabone'].to(self.device)
        self.real_ffa = input['ffa'].to(self.device)

        self.real_fi_mask = input['fi_mask'].to(self.device)
        self.real_fibone_mask = input['fibone_mask'].to(self.device)
        #self.real_ffabone_mask = input['ffabone_mask'].to(self.device)
        self.real_ffa_mask = input['ffa_mask'].to(self.device)
        # self.real_fihf = self.hfc_filter(self.real_fi,self.real_fi_mask)
        self.image_paths = input['fi_paths']

        # if self.isTrain:
            #pcg
            #self.rand = torch.rand(1)
            #if self.rand > self.opt.PTWH:
                # self.input6_A = torch.cat([self.real_A, self.real_A_HF], dim=1)
                # self.input6_B = torch.cat([self.real_B, self.real_B_HF], dim=1)        
            #else:
                #train with zero tensor
                # self.zero_tensor = torch.zeros_like(self.real_A)
                # self.input6_A = torch.cat([self.real_A, self.zero_tensor], dim=1)
                # self.input6_B = torch.cat([self.real_B, self.zero_tensor], dim=1)
        # else:
        #     self.zero_tensor = torch.zeros_like(self.real_A)
        #     self.input6_A = torch.cat([self.real_A, self.zero_tensor], dim=1)
        #     self.input6_B = torch.cat([self.real_B, self.zero_tensor], dim=1)


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # if self.isTrain:
        self.fake_ffa,self.fake_fibone = self.netG_fi2ffa(self.real_fi)
        # self.rec_fi,self.rec_fibone = self.netG_ffa2fi(self.fake_ffa)
        self.rec_fi,self.rec_fibone = self.netG_ffa2fi(self.fake_fibone)
        # self.fake_fi,self.fake_ffabone = self.netG_ffa2fi(self.real_ffa)
        # # self.rec_ffa,self.rec_ffabone = self.netG_fi2ffa(self.fake_fi)
        # self.rec_ffa,self.rec_ffabone = self.netG_fi2ffa(self.fake_ffabone)
        self.fake_fibone = mul_mask(self.fake_fibone,self.real_fibone_mask)
        self.fake_ffa = mul_mask(self.fake_ffa,self.real_fi_mask)
        self.rec_fi = mul_mask(self.rec_fi,self.real_fi_mask)
        self.rec_fibone = mul_mask(self.rec_fibone,self.real_fibone_mask)

        # self.fake_fi = mul_mask(self.fake_fi,self.real_ffa_mask)
        # self.fake_ffabone = mul_mask(self.fake_ffabone,self.real_ffabone_mask)
        # self.rec_ffa = mul_mask(self.rec_ffa,self.real_ffa_mask)
        # self.rec_ffabone = mul_mask(self.rec_ffabone,self.real_ffabone_mask)


    def D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        #loss_D.backward()
        return loss_D
    
    def backward_D_fibone(self):
        """
        Calculate high frequency loss for the discriminator, we want to closer rec_good'F and real_good'F
        """
        # fake_B_IC_IC = self.fake_B_IC_pool.query(self.fake_B_IC)
        self.loss_D_fibone = self.D_basic(self.netD_fibone, self.real_fibone, self.fake_fibone)

        # fake_B_IM_HF = self.fake_B_IM_HF_pool.query(self.fake_B_IM_HF)
        # self.loss_D_A_IC_HF = self.D_basic(self.netD_A_IC, self.real_A_HF, fake_B_IM_HF)

        # self.loss_D_A_IC = (self.loss_D_A_IC_IC + self.loss_D_A_IC_HF)*(1/2)*self.opt.lambda_A_D_HF*self.opt.adv_hf
        # self.loss_D_A_IC = (1.01+self.opt.TWHF)*self.loss_D_A_IC
        # self.loss_D_A_IC.backward()
        self.loss_D_fibone = self.loss_D_fibone*self.opt.lambda_A_D_fibone
        self.loss_D_fibone.backward()

    def backward_D_ffa(self):
        """Calculate GAN loss for discriminator D_A"""
        # fake_A_IM = self.fake_A_IM_pool.query(self.fake_A_IM)
        # self.loss_D_B_IM = self.D_basic(self.netD_B_IM, self.real_A, fake_A_IM)*self.opt.lambda_B_D_IM*self.opt.adv_im
        # self.loss_D_B_IM = (1.01+self.opt.TWHF)*self.loss_D_B_IM
        # self.loss_D_B_IM.backward()
        self.loss_D_ffa = self.D_basic(self.netD_ffa,self.real_ffa,self.fake_ffa)
        self.loss_D_ffa = self.loss_D_ffa*self.opt.lambda_A_D_ffa
        self.loss_D_ffa.backward()

    # def backward_D_ffabone(self):
    #     """Calculate GAN loss for discriminator D_A"""
    #     # fake_B_IM = self.fake_B_IM_pool.query(self.fake_B_IM)
    #     # self.loss_D_A_IM = self.D_basic(self.netD_A_IM, self.real_B, fake_B_IM)*self.opt.lambda_A_D_IM*self.opt.adv_im
    #     # self.loss_D_A_IM = (1.01+self.opt.TWHF)*self.loss_D_A_IM
    #     # self.loss_D_A_IM.backward()
    #     self.loss_D_ffabone = self.D_basic(self.netD_ffabone,self.real_ffabone,self.fake_ffabone)
    #     self.loss_D_ffabone = self.loss_D_ffabone*self.opt.lambda_B_D_ffabone
    #     self.loss_D_ffabone.backward()

    # def backward_D_fi(self):
    #     """Calculate GAN loss for discriminator D_B"""
    #     #self.loss_D_fi = self.D_basic(self.netD_fi, self.real_fi, self.fake_fi)
    #     self.loss_D_fi = self.D_basic(self.netD_fi, self.real_fi, self.rec_fi)
    #     self.loss_D_fi = self.loss_D_fi*self.opt.lambda_B_D_fi
    #     self.loss_D_fi.backward()        

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        # lambda_idt = self.opt.lambda_identity
        # lambda_A = self.opt.lambda_A
        # lambda_B = self.opt.lambda_B
        # Identity loss
        # if lambda_idt > 0:
        #     # G_A should be identity if real_B is fed: ||G_A(B) - B||
        #     self.idt_A,_ = self.netG_A(self.real_B)
        #     self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * self.opt.lambda_B_idt
        #     # G_B should be identity if real_A is fed: ||G_B(A) - A||
        #     self.idt_B,_ = self.netG_B(self.real_A)
        #     self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * self.opt.lambda_A_idt
        # else:
        #     self.loss_idt_A = 0
        #     self.loss_idt_B = 0
        # self.loss_idt = self.loss_idt_A + self.loss_idt_B


        #生成器对抗损失
        self.loss_DG_fibone = self.criterionGAN(self.netD_fibone(self.fake_fibone),True) * self.opt.lambda_A_DG_fibone
        self.loss_DG_ffa = self.criterionGAN(self.netD_ffa(self.fake_ffa),True) * self.opt.lambda_A_DG_ffa
        #self.loss_DG_ffabone = self.criterionGAN(self.netD_ffabone(self.fake_ffabone),True) * self.opt.lambda_B_DG_ffabone
        #self.loss_DG_fi = self.criterionGAN(self.netD_fi(self.fake_fi),True) * self.opt.lambda_B_DG_fi
        #self.loss_DG_fi = self.criterionGAN(self.netD_fi(self.rec_fi),True) * self.opt.lambda_B_DG_fi
        #self.loss_G_adv = self.loss_DG_fibone+self.loss_DG_ffabone+self.loss_DG_fi+self.loss_DG_ffa
        self.loss_G_adv = self.loss_DG_fibone+self.loss_DG_ffa
        #正向损失

        self.loss_G_A_ffa = self.criterionCycle(self.fake_ffa,self.real_ffa) * self.opt.lambda_A_G_ffa
        self.loss_G_A_fibone1 = self.criterionCycle(self.fake_fibone,self.real_fibone) * self.opt.lambda_A_G_fibone1
        self.loss_G_A_fi = self.criterionCycle(self.rec_fi,self.real_fi) * self.opt.lambda_A_G_fi
        self.loss_G_A_fibone2 = self.criterionCycle(self.rec_fibone,self.real_fibone) * self.opt.lambda_A_G_fibone2
        self.loss_G_A = self.loss_G_A_fibone1 + self.loss_G_A_ffa + self.loss_G_A_fi + self.loss_G_A_fibone2
        #self.loss_G_A = self.loss_G_A_fihf + self.loss_G_A_fi + self.loss_G_A_ffa
        #self.loss_G_A =  self.loss_G_A_fi + self.loss_G_A_fibone2

        #反向损失
        # self.loss_G_B_fi = self.criterionCycle(self.fake_fi,self.real_fi) * self.opt.lambda_B_G_fi
        # self.loss_G_B_ffabone1 = self.criterionCycle(self.fake_ffabone,self.real_ffabone) * self.opt.lambda_B_G_ffabone1
        # self.loss_G_B_ffabone2 = self.criterionCycle(self.rec_ffabone,self.real_ffabone) * self.opt.lambda_B_G_ffabone2
        # self.loss_G_B_ffa = self.criterionCycle(self.rec_ffa,self.real_ffa) * self.opt.lambda_B_G_ffa
        # self.loss_G_B = self.loss_G_B_fi + self.loss_G_B_ffabone1 + self.loss_G_B_ffabone2 + self.loss_G_B_ffa
        #self.loss_G_B = self.loss_G_B_ffa + self.loss_G_B_ffabone2 
       
        #合并运算
        # self.loss_G = self.loss_G_adv + self.loss_G_A + self.loss_G_B
        self.loss_G = self.loss_G_adv + self.loss_G_A
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        #self.set_requires_grad([self.netD_fibone,self.netD_ffabone,self.netD_fi,self.netD_ffa], False)  # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.netD_fibone,self.netD_ffa], False)         
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        #self.set_requires_grad([self.netD_fibone,self.netD_ffabone,self.netD_fi,self.netD_ffa], True)
        self.set_requires_grad([self.netD_fibone,self.netD_ffa], True)        
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_fibone()      # calculate gradients for D_A
        # self.backward_D_ffabone()      # calculate gradients for D_A
        # self.backward_D_fi()      # calculate graidents for D_B
        self.backward_D_ffa()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights


def mul_mask(image, mask):
    return (image + 1) * mask - 1