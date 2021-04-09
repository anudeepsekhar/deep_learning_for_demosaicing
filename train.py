#%%
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import get_data_loaders
from model import UNet

from tqdm import tqdm
from collections import defaultdict


#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calc_loss(outputs, labels, metrics):
    criterion = nn.MSELoss()
    loss = criterion(outputs, labels)

    metrics['loss'] += loss.data.cpu().numpy() * labels.shape[0]
    return loss

def create_masks(batch_size, Nc, Ny, Nx):
        
        mask_original= np.zeros([batch_size, Nc, Ny, Nx])
        for b in range(batch_size):
            R = np.zeros([Ny, Nx])
            G = np.zeros([Ny, Nx])
            B = np.zeros([Ny, Nx]) 
            for i in range(1,Ny//2):
                for j in range(1,Nx//2):
                    R[2*i-1,2*j-1] = True
                    B[2*i,2*j] = True
                    G[2*i,2*j-1] = True
                    G[2*i-1,2*j] = True
            mask_original[b] = np.array([R,G,B])



        # mask_interm = np.bitwise_not(mask_original)
        mask_interm = np.invert(mask_original.astype(np.bool))
                
        return torch.Tensor(mask_original.astype(np.float32)), torch.Tensor(mask_interm.astype(np.float32))

def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))  


def train_model(model, optimizer, num_epochs=25):
    for epoch in (range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)


        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, targets in dataloaders[phase]:
                inputs = inputs.to(device)
                targets = targets.to(device)
                # print(inputs.shape)
                batch_size, Nc, Ny, Nx = inputs.shape
                mask_original, mask_interm = create_masks(batch_size, Nc, Ny, Nx)
                mask_interm = mask_interm.to(device)
                mask_original = mask_original.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #first pass
                    interm = model(inputs)
                    #TODO: implement the masking logic
                    # mask out original values in interm
                    interm_input = torch.mul(interm,mask_original)
                    final = model(interm_input)
                    # mask out interm values
                    final = torch.mul(interm,mask_interm)
                    # Calculate L2 loss
                    loss = calc_loss(final, targets, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

                print_metrics(metrics, epoch_samples, phase)
                epoch_loss = metrics['loss'] / epoch_samples

#%%
dataloaders = get_data_loaders()
model = UNet(n_class=3)
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
train_model(model, optimizer)

                
# %%
