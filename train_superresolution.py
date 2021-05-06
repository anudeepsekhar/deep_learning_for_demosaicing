#%%
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time
import pdb

from model_utils import *
from dataset import get_CUB200_loader
from model import ResidualBlock_Superresolution,Net_Superresolution
from training import train_model

from tqdm import tqdm
import os
import math
from PIL import Image

import pickle
import argparse

resume_from_ckp = False
trialNumber = 12
checkpoint_path = "./checkpoint_superresolution/"+"trial"+str(trialNumber)+"checkpoint.pt"
num_epochs = 3
withRedNet = False
withSRCNN = False

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_ckp(state, checkpoint_dir):
    torch.save(state, checkpoint_dir)


# def train_model(model, optimizer, scheduler, num_epochs,start_epoch,checkpoint_dir):

#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_loss = 1e10

#     for epoch in (range(start_epoch,num_epochs)):
#         print('\n Epoch {}/{}'.format(epoch+1, num_epochs))
#         print('-' * 10)

#         since = time.time()

#         # train
#         model.train()
#         print("Training...")
#         training_loss = 0.0
            
#             # epoch_samples = 0

#         for inputs, targets in tqdm(dataloaders['train']):
#             # zero the parameter gradients
#             optimizer.zero_grad()
#             inputs = inputs.to(device)
#             targets = targets.to(device)

#             result = model(inputs)
#             # print('result.shape: ',result.shape)

#             # Calculate loss
#             criterion = nn.MSELoss()
#             loss = criterion(result.float(), targets.float())
 
#             loss.backward()
#             optimizer.step()

#             training_loss += loss.item()

#         num_batch = len(dataloaders['train'])
#         training_loss /= num_batch

#         checkpoint = {
#         'epoch': epoch + 1,
#         'state_dict': model.state_dict(),
#         'optimizer': optimizer.state_dict(),
#         'scheduler': scheduler.state_dict()
#         }
#         # saves the checkpoint for resuming in case instance disconnected

#         save_ckp(checkpoint, checkpoint_dir)

#         print('Epoch: {}, Average training loss: {:.6f}'.format(epoch+1,training_loss))
        
#         # validate
#         print("Validating...")
#         model.eval()
#         val_loss = 0.0

#         for inputs, targets in tqdm(dataloaders['val']):
#           with torch.no_grad():
#             inputs = inputs.to(device)
#             targets = targets.to(device)

#             result = model(inputs)

#             # Calculate loss
#             criterion = nn.MSELoss()
#             loss = criterion(result.float(), targets.float())

#             val_loss += loss.item()
#             # statistics

#         num_batch = len(dataloaders['val'])
#         val_loss /= num_batch

#         print('Epoch: {}, Average validating loss: {:.6f}'.format(epoch+1,val_loss))
#         scheduler.step(val_loss)


#         # deep copy the model
#         if val_loss < best_loss:
#             print("saving best model")
#             best_loss = val_loss
#             best_model_wts = copy.deepcopy(model.state_dict())

#         time_elapsed = time.time() - since
#         print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#     print('Best val loss: {:6f}'.format(best_loss))

#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     return model



#%%
if __name__ == "__main__":

  dataloaders = get_CUB200_loader()

  # save dataloaders to file for use in testing script
  # Open a file and use dump()
  var_save_dir = './data/variables'
  var_name = 'dataloaders'+'_trial'+str(trialNumber)+'.pkl'
  path = var_save_dir + '/' + var_name
  if not os.path.exists(var_save_dir):
    os.makedirs(var_save_dir)
  with open(path, 'wb') as file:
    # A new file will be created
    pickle.dump(dataloaders, file)

  lr = 1e-4
  weightDecay = 5e-5
  model = Net_Superresolution(withRedNet=withRedNet,withSRCNN=withSRCNN)
  model = model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weightDecay)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=1,factor=0.1,verbose=True)

  if resume_from_ckp:
    ckp_path = checkpoint_path
    ckp = torch.load(ckp_path, map_location=device)
    model.load_state_dict(ckp['state_dict'])
    optimizer.load_state_dict(ckp['optimizer'])
    scheduler.load_state_dict(ckp['scheduler'])
    start_epoch = ckp['epoch']
    print("Resuming from checkpoint...")
    del ckp
  else:
    start_epoch = 0
  
  if not os.path.exists('checkpoint_superresolution'):
    os.makedirs('checkpoint_superresolution')
  
  model = train_model(model, optimizer, scheduler, num_epochs=num_epochs, start_epoch=start_epoch,checkpoint_dir=checkpoint_path,dataloaders=dataloaders)
  filename = "./model/"+"trial"+str(trialNumber)+".pth"
  torch.save(model.state_dict(), filename)


# %%
