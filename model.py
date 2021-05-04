#%%
import torch
import torch.nn as nn
from torchsummary import summary
import math

#%%
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   


class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out

# for resnet
class SimpleResidualBlock(nn.Module):
    def __init__(self, in_channel_size, out_channel_size, identity_downsample=None, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channel_size, out_channel_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel_size)
        self.conv2 = nn.Conv2d(out_channel_size, out_channel_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel_size)
        
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        # print("identity:", identity)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.stride!=1:
            # print("self.stride is not 1, self.identity_downsample applied")
            identity = self.identity_downsample(x)

        # print("out shape", out.shape)
        # print("identity shape", identity.shape)
        out += identity
        out = self.relu(out)    
        # shortcut = self.shortcut(x)
        
        # out = self.relu(out + shortcut)
        
        return out

# resnet 34
# Reference: https://gist.github.com/nikogamulin/7774e0e3988305a78fd73e1c4364aded
class Resnet34(nn.Module):
    def __init__(self, block, in_features):
        super().__init__()
        layers = [3, 4, 6, 3]
        # class attributes
        self.in_channels = 64
        self.expansion = 1

        self.conv1 = nn.Conv2d(in_features, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # resnet layers
        self.layer1 = self.make_layers(block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layers(block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layers(block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layers(block, layers[3], intermediate_channels=512, stride=2)

        # another extra conv layer to correct the channel to 3
        self.conv2 = nn.Conv2d(512,3,kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(3)

        self.avgpool = nn.AdaptiveAvgPool2d((32, 32))
        # self.flattenlayer = nn.Flatten()
        # self.dropoutlayer = nn.Dropout(p=0.3)
        # self.fc = nn.Linear(512,num_classes) 

           
    def forward(self, x):
        # print("input shape: ",x.shape)
        x = self.conv1(x)
        # print("x after self.conv1 shape: ",x.shape)
        x = self.bn1(x)
        # print("x after self.bn1 shape: ",x.shape)
        x = self.relu(x)
        # print("x after self.relu shape: ",x.shape)
        x = self.maxpool(x)
        # print("x after self.maxpool shape: ",x.shape)
        # x = self.dropoutlayer(x)
        x = self.layer1(x)
        # print("x after self.layer1 shape: ",x.shape)
        x = self.layer2(x)
        # print("x after self.layer2 shape: ",x.shape)
        x = self.layer3(x)
        # print("x after self.layer3 shape: ",x.shape)
        x = self.layer4(x)
        # print("x after self.layer4 shape: ",x.shape)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # print("x after self.conv2 shape: ",x.shape)

        output = self.avgpool(x)

        # embedding = self.flattenlayer(x)
        # embedding = self.dropoutlayer(embedding)
        # output = self.fc(embedding)
        # return output,embedding

        return output
 
    def make_layers(self, block, num_residual_blocks, intermediate_channels, stride):
        layers = []
        downsample = None
        if stride != 1 or self.in_channels!=intermediate_channels*self.expansion:
          downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(intermediate_channels*self.expansion))
        layers.append(block(self.in_channels, intermediate_channels,identity_downsample=downsample,stride=stride))
        self.in_channels = intermediate_channels * self.expansion # 256 # next residual block's in_channel will be the block's typical channel number
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels)) # 256 -> 64, 64*4 (256) again
        return nn.Sequential(*layers)

# Joint demosaicking and superresolution
# the following code was taken from https://github.com/GitHberChen/Deep-Residual-Network-for-JointDemosaicing-and-Super-Resolution/blob/master/Model.py
class ResidualBlock_Superresolution(nn.Module):
    def __init__(self):
        super(ResidualBlock_Superresolution, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256), # try adding bn
            nn.PReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256), # try adding bn
        )
        self.shortcut = nn.Sequential()
        self.active_f = nn.PReLU()

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = self.active_f(out)
        return out


class Net_Superresolution(nn.Module):

    def __init__(self, resnet_level=24, withRedNet=False, withSRCNN=False):
        super(Net_Superresolution, self).__init__()

        # ***Stage1***
        # class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.stage1_1_conv4x4 = nn.Conv2d(in_channels=1, out_channels=256,
                                          kernel_size=4, stride=2, padding=1, bias=True)
        # Reference:
        # CLASS torch.nn.PixelShuffle(upscale_factor)
        # Examples:
        #
        # >>> pixel_shuffle = nn.PixelShuffle(3)
        # >>> input = torch.randn(1, 9, 4, 4)
        # >>> output = pixel_shuffle(input)
        # >>> print(output.size())
        # torch.Size([1, 1, 12, 12])
        # self.shortcut = nn.Sequential()
        self.stage1_2_SP_conv = nn.PixelShuffle(2)
        self.stage1_2_conv4x4 = nn.Conv2d(in_channels=64, out_channels=256,
                                          kernel_size=3, stride=1, padding=1, bias=True)

        # CLASS torch.nn.PReLU(num_parameters=1, init=0.25)
        self.stage1_2_PReLU = nn.PReLU()

        # ***Stage2***
        self.stage2_ResNetBlock = []
        for i in range(resnet_level):
            self.stage2_ResNetBlock.append(ResidualBlock_Superresolution())
        self.stage2_ResNetBlock = nn.Sequential(*self.stage2_ResNetBlock)

        # ***Stage3***
        self.stage3_1_SP_conv = nn.PixelShuffle(2)
        self.stage3_2_conv3x3 = nn.Conv2d(in_channels=64, out_channels=256,
                                          kernel_size=3, stride=1, padding=1, bias=True)
        self.stage3_2_PReLU = nn.PReLU()
        self.stage3_3_conv3x3 = nn.Conv2d(in_channels=256, out_channels=3,
                                          kernel_size=3, stride=1, padding=1, bias=True)
        
        # RedNet
        self.withRedNet = withRedNet
        if withRedNet:
          self.rednet20 = REDNet20()

        # SRCNN
        self.withSRCNN = withSRCNN
        if withSRCNN:
          self.srcnn = SRCNN(num_channels=3)

    def forward(self, x):
        out = self.stage1_1_conv4x4(x)
        out = self.stage1_2_SP_conv(out)
        out = self.stage1_2_conv4x4(out)
        # out_skip = self.shortcut(out)
        out = self.stage1_2_PReLU(out)

        out = self.stage2_ResNetBlock(out)

        # # try skip connect here
        # out += out_skip

        out = self.stage3_1_SP_conv(out)
        out = self.stage3_2_conv3x3(out)
        out = self.stage3_2_PReLU(out)
        out = self.stage3_3_conv3x3(out)

        # rednet (if applicable)
        if self.withRedNet:
          out = self.rednet20(out)

        # SRCNN (if applicable)
        if self.withSRCNN:
          out = self.srcnn(out)

        return out

# this performs upsampling by scale factor 2, then pass it through REDNet20
class REDNet_model(nn.Module):
  def __init__(self):
    super().__init__()
    self.rednet = REDNet20()
    self.upsample = nn.Upsample(scale_factor=2,mode='bicubic')

  def forward(self,x):
    out = self.upsample(x)
    out = self.rednet(out)
    return out

# this code is taken from https://github.com/yjn870/REDNet-pytorch/blob/11ee46722a4fbee48b37f417e4329026d5b78bfa/model.py#L35
class REDNet20(nn.Module):
    def __init__(self, num_layers=10, num_features=64):
        super(REDNet20, self).__init__()
        self.num_layers = num_layers

        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(3, num_features, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(inplace=True)))
        for i in range(num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True)))

        for i in range(num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.ConvTranspose2d(num_features, 3, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(x)

        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                x = x + conv_feat
                x = self.relu(x)

        x += residual
        x = self.relu(x)

        return x

# code taken from https://github.com/yjn870/SRCNN-pytorch/blob/master/models.py
class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
model = UNet(n_class=3)
model = model.to(device)
# %%
summary(model, input_size=(3, 64, 64))
# %%
