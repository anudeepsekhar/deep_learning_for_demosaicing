#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as data

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
        B = np.zeros([2*Ny, 2*Nx])
        for i in range(1,Ny):
            for j in range(1,Nx):
                B[2*i-1,2*j-1] = img[0,i,j]
                B[2*i,2*j] = img[2,i,j]
                B[2*i,2*j-1] = img[1,i,j]
                B[2*i-1,2*j] = img[1,i,j]

        return B

    def __call__(self, pic):
        
        return self.remosaic(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'
#%%
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        ToBayer()

     ])

class CIFAR10MosaicDataset(data.CIFAR10):
    def __init__(self, root, train, download, transform=None):
        super(CIFAR10MosaicDataset, self).__init__(root=root, train=train, download=download, transform=transform)
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
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, img


    

#%%
trainset = CIFAR10MosaicDataset(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = CIFAR10MosaicDataset(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
def get_data_loaders():
    return trainloader, testloader
