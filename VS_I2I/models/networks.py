import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from scipy import ndimage
from skimage import color, data, restoration

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, 
             n_blocks_local=3, norm='instance', gpu_ids=[]):    
    norm_layer = get_norm_layer(norm_type=norm)     
    if netG == 'global':    
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)       
    elif netG == 'local':        
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, 
                                  n_local_enhancers, n_blocks_local, norm_layer)
    elif netG == 'encoder':
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    else:
        raise('generator not implemented!')
    print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):        
    norm_layer = get_norm_layer(norm_type=norm)   
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)   
    print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class DiceBCELoss(nn.Module):               #BCE-Dice Loss
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

# class BinarySegLoss(nn.Module):               #Binary_Segmentation
#     def __init__(self, device,requires_grad=False):
#         super(BinarySegLoss, self).__init__()
#         self.device = device
#         print("****** Using Binary Segmentation Loss *********")

#     def __call__(self, inputs, targets):
#         if not torch.is_tensor(inputs):
#             raise TypeError("Input x type is not a torch.Tensor. Got {}"
#                             .format(type(inputs)))
#         if not len(inputs.shape) == 4:
#             raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
#                              .format(inputs.shape))
        
#         true_B = targets.clone().detach()
#         input_B = inputs.clone().detach()

#         mask_GT = torch.relu(torch.sign(torch.sigmoid(input_B)-0.5))
#         mask_target = torch.relu(torch.sign(torch.sigmoid(true_B)-0.5))
#         weighted = (mask_GT * mask_target).sum()

#         loss = torch.abs(inputs-targets)

#         weighted.requires_grad = False
#         return torch.mul(loss,weighted).mean() 

class GaussianBlurringLoss(nn.Module):               #Gaussian-Blurring Weighted Kernel Loss
    def __init__(self, device,requires_grad=False):
        super(GaussianBlurringLoss, self).__init__()
        self.device = device
        print("****** Using Gaussian Blurring Loss *********")       #Sigma of 1 for 5 x 5 kernel initialization
        self.weights =  torch.tensor([                                               
                           [1.,4.,7.,4.,1.],
                           [4.,16.,26.,16.,4.],
                           [7.,26.,41.,26.,7.],
                           [4.,16.,26.,16.,4.],
                           [1.,4.,7.,4.,1.]])
                        
        self.weights = self.weights.view(1,1,5,5).repeat(3,1,1,1)
        self.weights = self.weights.cuda()
        self.weights = self.weights.to(self.device[0])

    def __call__(self, inputs, targets):
        if not torch.is_tensor(inputs):
            raise TypeError("Input x type is not a torch.Tensor. Got {}"
                            .format(type(inputs)))
        if not len(inputs.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(inputs.shape))
        ################################# Weighted Kernel Mapping #######################################
        loss = torch.abs(inputs-targets)
        true_B = targets.clone().detach()
        weighted = F.conv2d(true_B,self.weights,padding = 2, stride =1, bias = None, groups=3)
        weighted = (weighted - torch.min(weighted))/273.*(torch.max(weighted) - torch.min(weighted))
        weighted.requires_grad = False
        return torch.mul(loss,weighted).mean()  

class IoULoss(nn.Module):             #Jaccard/Intersection over Union (IoU) Loss
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = 1 - (intersection + smooth)/(union + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        IoULoss = IoU + BCE        
        return IoULoss

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, alpha=2, gamma=1):
        super(FocalLoss, self).__init__()
        self.alpha=alpha
        self.gamma=gamma

    def forward(self, inputs, targets, alpha=None, gamma=None, smooth=1):
        if alpha:
            self.alpha = alpha
        if gamma:
            self.gamma = gamma
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE
                       
        return focal_loss

class KernelConvLoss(nn.Module):
    def __init__(self, device,requires_grad=False):
        super(KernelConvLoss, self).__init__()
        self.device = device
        print("****** Using PSF Kernel Convolution Loss *********")
        self.weights =  torch.tensor([                                #Proposed Dataset: Pixel_Size -80, 609nm, NA= 1.4                 
                           [5.,35.,55.,35.,5.],
                           [35.,125.,181.,125.,25.],
                           [55.,181.,255.,181.,55.],
                           [35.,125.,181.,125.,25.],
                           [5.,35.,55.,35.,5.]])
                        
        self.weights = self.weights.view(1,1,5,5).repeat(3,1,1,1)
        self.weights = self.weights.cuda()
        self.weights = self.weights.to(self.device[0])

    def __call__(self, inputs, targets):
        if not torch.is_tensor(inputs):
            raise TypeError("Input x type is not a torch.Tensor. Got {}"
                            .format(type(inputs)))
        if not len(inputs.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(inputs.shape))
        ####################################################################################
        loss = torch.abs(inputs-targets)
        true_B = targets.clone().detach()
        weighted = F.conv2d(true_B,self.weights,padding = 2, stride =1, bias = None, groups=3)
        weighted = (weighted - torch.min(weighted))/(torch.max(weighted) - torch.min(weighted))
        weighted.requires_grad = False
        return torch.mul(loss,weighted).mean()  


class KernelLoss(nn.Module):
    def __init__(self, device,requires_grad=False):
        super(KernelLoss, self).__init__()
        self.device = device
        print("****** Using PSF Kernel Deconvolution Loss *********")
        self.weights =  torch.tensor([                                #Proposed Dataset: Pixel_Size -80, 609nm, NA= 1.4                 
                           [5.,35.,55.,35.,5.],
                           [35.,125.,181.,125.,25.],
                           [55.,181.,255.,181.,55.],
                           [35.,125.,181.,125.,25.],
                           [5.,35.,55.,35.,5.]])

        # elif use_mito:
        #     print("****** Using Ounkomol et al. (2018) Mito_ER Dataset PSF Kernel *********")
        #self.weights =  torch.tensor([                               #Pixel_Size -108nm, Lamda- 561nm, NA= 1.25
        #                    [2.,3.,12.,3.,2],
        #                    [3.,71.,140.,71.,3],
        #                    [12.,140.,255.,140.,12],
        #                    [3.,71.,140.,71.,3],
        #                    [2.,3.,12.,3.,2.]])

        # elif use_membrane:
        #     print("****** Using Ounkomol et al. (2018) Membrane Dataset PSF Kernel *********")
        # self.weights =  torch.tensor([                               #Cell_Membrane: Pixel_Size -86nm, Lamda- 561nm, NA= 1.2
        #                   [5.,34.,55.,34.,5],
        #                   [34.,125.,181.,125.,34],
        #                   [55.,181.,255.,181.,55],
        #                   [34.,125.,181.,125.,34],
        #                   [5.,34.,55.,34.,5.]])

        # else:
        #     print("****** Invalid PSF Kernel *********")

        self.weights = self.weights.view(1,1,5,5).repeat(3,1,1,1)
        self.weights = self.weights.cuda()
        self.weights = self.weights.to(self.device[0])

    def __call__(self, inputs, targets):
        if not torch.is_tensor(inputs):
            raise TypeError("Input x type is not a torch.Tensor. Got {}"
                            .format(type(inputs)))
        if not len(inputs.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(inputs.shape))
        #######################################################################################################
        true_B = inputs.clone().detach()
        weighted = restoration.richardson_lucy(true_B.numpy(), self.weights.numpy(), iterations=20).cuda()
        weighted = (weighted - torch.min(weighted))/(torch.max(weighted) - torch.min(weighted))
        loss = torch.abs(weighted - targets)
        return loss.mean()
        ###############################################################################################################

class ThresholdLoss(nn.Module):
    def __init__(self,device,requires_grad=False):
        super(ThresholdLoss, self).__init__()
        self.device = device

    def __call__(self, inputs, targets):
        gt = targets.clone().detach()
        Min, Mean, Max, sd = torch.min(gt[0][0]), torch.mean(gt[0][0]), torch.max(gt[0][0]), torch.std(gt[0][0])
        loss = torch.abs(targets - inputs)

        M1 = Mean + sd
        M2 = Mean + 3 * sd
       
        for i in range(gt.shape[2]):
            for j in range(gt.shape[3]):
                if (gt[:, :, i, j] <= M1):       # Normalised Mean to Mean+ S.D -> [0.0,0.20]
                    gt[:, :, i, j] = ((gt[:, :, i, j] - Min)/ (M1 - Min)) * 0.20 
                    
                elif (gt[:, :, i, j] > M1  and gt[:, :, i, j] <= M2):       # Normalised Mean + SD to Mean+ 3*S.D -> [0.20,0.70]
                    gt[:, :, i, j] = ((gt[:, :, i, j] - M1)/ (M2 - M1)) * 0.50  + 0.2
                    
                else:
                    gt[:, :, i, j] = ((gt[:, :, i, j] - M2)/ (Max - M2)) * 0.30  + 0.70 # Normalised Mean + S.D. to Max -> [0.70,1.0]  
        return torch.mul(loss,gt).mean()

# Care must be taken when writing loss functions for PyTorch. 
# If you call a function to modify the inputs that doesn't entirely 
# use PyTorch's numerical methods, the tensor will 'detach' from the 
# graph that maps it back through the neural network for the purposes 
# of backpropagation, making the loss function unusable.

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, use_dice_bce= False, use_iou= False, use_focal=False, 
                        target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        
        if use_dice_bce:
            print("****** Using DiceBCE Loss for GAN_Loss *********")
            self.loss = DiceBCELoss()
        elif use_focal:
            print("****** Using Focal Loss for GAN_Loss *********")
            self.loss = FocalLoss()
        elif use_iou:
            print("****** Using IOU/Jaccard Loss for GAN_Loss *********")
            self.loss = IoULoss()
        elif use_lsgan:
            print("****** Using MSE Loss for GAN_Loss *********")
            self.loss = nn.MSELoss()
        elif use_lsgan:
            print("****** Using MSE Loss for GAN_Loss *********")
            self.loss = nn.MSELoss()
        else:
            print("****** Using BCE Loss for GAN_Loss *********")
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=7, 
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):        
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        
        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model        
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers        
        self.model = nn.Sequential(*model_global)                

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample            
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=5, stride=2, padding=2), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=5, stride=2, padding=2, output_padding=1), 
                               norm_layer(ngf_global), nn.ReLU(True)]      

            ### final convolution
            if n == n_local_enhancers:                
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]                       
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))                  
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input): 
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])        
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=7, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=5, stride=2, padding=2),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=5, stride=2, padding=2, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        return self.model(input)             
        
# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=3, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()        
        self.output_nc = output_nc        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), 
                 norm_layer(ngf), nn.ReLU(True)]             
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=5, stride=2, padding=2),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=5, stride=2, padding=2, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]        

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model) 

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))        
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b+1] == int(i)).nonzero() # n x 4            
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]                    
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)                                        
                    outputs_mean[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat                       
        return outputs_mean

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=4, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=4, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
