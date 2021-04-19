#%%
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import time
import pdb

from model_utils import *
from dataset import get_data_loaders
from model import UNet

from tqdm import tqdm
from collections import defaultdict
import os

resume_from_ckp = True
trialNumber = 4
checkpoint_path = "./checkpoint/"+"trial"+str(trialNumber)+"checkpoint.pt"
num_epochs = 4

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_ckp(state, checkpoint_dir):
    f_path = checkpoint_path
    torch.save(state, f_path)


def train_model(model, optimizer, num_epochs,start_epoch,checkpoint_dir):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in (range(start_epoch,num_epochs)):
        print('\n Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        since = time.time()

        # train
        model.train()
        print("Training...")
        training_loss = 0.0
            
        metrics = defaultdict(float)
            # epoch_samples = 0
        mask_is_created = False
        for inputs, targets in tqdm(dataloaders['train']):
            # zero the parameter gradients
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)

            batch_size, Nc, Ny, Nx = inputs.shape

            # forward
            
            # pass through UNet
            model_output = model(inputs)
            
            # create mask once
            if not mask_is_created:
              mask = create_RGB_masks(inputs[0]) # create a mask for a single image
              mask_is_created = True
            # In the following: mask.unsqueeze(0) turns mask from an [C, M, N] to [1, C, M, N] 
            # and .repeat(batch_size,1, 1, 1) repeats the tensor batch_size times along the zeroth dimension.
            mask_expanded = mask.unsqueeze(0).repeat(batch_size,1, 1, 1) 
            mask_expanded = mask_expanded.to(device)

            # apply mask to the batch of model_output and targets
            result = model_output*mask_expanded
            targets = targets*mask_expanded

            # Calculate L2 loss
            criterion = nn.MSELoss()
            loss = criterion(result.float(), targets.float())
 
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            # statistics

        num_batch = len(dataloaders['train'])
        training_loss /= num_batch

        checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        # 'scheduler': scheduler.state_dict()
        }
        # saves the checkpoint for resuming in case instance disconnected

        save_ckp(checkpoint, checkpoint_dir)

        # print_metrics(metrics, num_batch, 'train')
        print('Epoch: {}, Average training loss: {:.6f}'.format(epoch+1,training_loss))
        
        # validate
        print("Validating...")
        model.eval()
        val_loss = 0.0

        mask_is_created = False
        for inputs, targets in tqdm(dataloaders['val']):
          with torch.no_grad():
            inputs = inputs.to(device)
            targets = targets.to(device)

            batch_size, Nc, Ny, Nx = inputs.shape

            # forward
            
            # pass through UNet
            model_output = model(inputs)

            # create mask once
            if not mask_is_created:
              mask = create_RGB_masks(inputs[0]) # create a mask for a single image
              mask_is_created = True
            # In the following: mask.unsqueeze(0) turns mask from an [C, M, N] to [1, C, M, N] 
            # and .repeat(batch_size,1, 1, 1) repeats the tensor batch_size times along the zeroth dimension.
            mask_expanded = mask.unsqueeze(0).repeat(batch_size,1, 1, 1) 
            mask_expanded = mask_expanded.to(device)

            # apply mask to the batch of model_output and targets
            result = model_output*mask_expanded
            targets = targets*mask_expanded

            # Calculate L2 loss
            criterion = nn.MSELoss()
            loss = criterion(result.float(), targets.float())

            val_loss += loss.item()
            # statistics

        num_batch = len(dataloaders['val'])
        val_loss /= num_batch

        print('Epoch: {}, Average validating loss: {:.6f}'.format(epoch+1,val_loss))


        # deep copy the model
        if val_loss < best_loss:
            print("saving best model")
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:6f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

#%%
if __name__ == "__main__":
  dataloaders = get_data_loaders()
  model = UNet(n_class=3)
  model = model.to(device)
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

  if resume_from_ckp:
    ckp_path = checkpoint_path
    ckp = torch.load(ckp_path, map_location=device)
    model.load_state_dict(ckp['state_dict'])
    optimizer.load_state_dict(ckp['optimizer'])
    # scheduler.load_state_dict(ckp['scheduler'])
    start_epoch = ckp['epoch']
    print("Resuming from checkpoint...")
    del ckp
  else:
    start_epoch = 0
  
  if not os.path.exists('checkpoint'):
    os.makedirs('checkpoint')
  
  model = train_model(model, optimizer, num_epochs=num_epochs, start_epoch=start_epoch,checkpoint_dir=checkpoint_path)
  if not os.path.exists('model'):
    os.makedirs('model')
  filename = "./model/"+"trial"+str(trialNumber)+".pth"
  torch.save(model.state_dict(), filename)
                
# %%
