#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as data
from torchvision.transforms import ToPILImage 

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import os.path
import numpy as np
import pickle
#%%

class ToBayer(object):
    def remosaic(self,img):
        Nc, Ny, Nx = img.shape
        R = np.zeros([2*Ny, 2*Nx])
        G = np.zeros([2*Ny, 2*Nx])
        B = np.zeros([2*Ny, 2*Nx])
        # R_mask = -1*np.ones([2*Ny, 2*Nx])
        for i in range(1,Ny):
            for j in range(1,Nx):
                R[2*i-1,2*j-1] = img[0,i,j]
                # R_mask[2*i-1,2*j-1] = 0
                B[2*i,2*j] = img[2,i,j]
                G[2*i,2*j-1] = img[1,i,j]
                G[2*i-1,2*j] = img[1,i,j]



        return np.array([R,G,B])

    def __call__(self, pic):
        
        return self.remosaic(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'

# resample the original RGB image into four subsamples
class Resampling(object):
    # def __init__(self,img):
    #   self.img = img

    def resample(self,img):
        Nc, Ny, Nx = img.shape
        R = np.zeros([Ny, Nx])
        G = np.zeros([Ny, Nx])
        B = np.zeros([Ny, Nx])

        # green channel
        for i in range(0,Ny,1): #row
          for j in range(0,Nx,2): #column
            if (i%2)==0: # even rows
              G[i,j+1] = img[1,i,j+1]
            elif (i%2)==1: # odd rows
              G[i,j] = img[1,i,j]

        # red channel and blue channel
        for i in range(0,Ny,2):
          for j in range(0,Nx,2):
            B[i,j] = img[2,i,j] # blue channel
            R[i+1,j+1] = img[0,i+1,j+1] # red channel
        
        # self.resampled = np.array([R,G,B])
        # print("RGB resampled: ", np.array([R,G,B]))
        return torch.from_numpy(np.array([R,G,B]))

    def __call__(self, img):
        
        return self.resample(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'

#%%
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        Resampling()
        # ToBayer()

     ])

transform2 = transforms.Compose(
    [
        transforms.ToTensor()

     ])

class CIFAR10MosaicDataset(data.CIFAR10):
    def __init__(self, root, train, download, transform=None):
        super(CIFAR10MosaicDataset, self).__init__(root=root, train=train, download=download, transform=transform)
        if not os.path.exists('./data/transformed_cifar'):
          os.makedirs('./data/transformed_cifar')
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img_original = Image.fromarray(img) 
        
        # print(np.amax(img))
        if self.transform is not None:
          img = self.transform(img_original)
        
        img_original = transform2(img_original)

        if self.target_transform is not None:
            target = self.target_transform(target)

        
        # img1 = (img * 255).astype(np.uint8) # scale it back to 0-255 and save
        # file_name = 'cifar_'+str(index)+'.png'
        # dir_path = './data/transformed_cifar'+'/'+file_name
        # if not os.path.exists(dir_path):
        #   img1 = ToPILImage()(img)
        #   img1.save(dir_path,"PNG")
        
        # np.save(dir_path,img1)
        # return img.astype(np.float32), img.astype(np.float32)
        return img.float(), img_original.float()


    

#%%
trainset = CIFAR10MosaicDataset(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

testset = CIFAR10MosaicDataset(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)



def get_data_loaders():
    return {'train':trainloader, 'val':testloader}



# %%
for batch in trainloader:
    inputs, targets = batch
    print(inputs.shape)
    print(targets.shape)
    plt.imshow(inputs[0,0,:])
    break
# %%
