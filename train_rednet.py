#%%
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time
import pdb


from dataset import get_CUB200_loader
from model import REDNet_model
from training import train_model

from tqdm import tqdm
import os
import math
from PIL import Image

import pickle
import argparse

resume_from_ckp = False
trialNumber = 8
checkpoint_path = "./checkpoint_rednet/"+"trial"+str(trialNumber)+"checkpoint.pt"
num_epochs = 3

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
if __name__ == "__main__":
  dataloaders = get_CUB200_loader(three_channel_bayer=True)

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
  model = REDNet_model()

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
  
  if not os.path.exists('checkpoint_rednet'):
    os.makedirs('checkpoint_rednet')
  
  model = train_model(model, optimizer, scheduler, num_epochs=num_epochs, start_epoch=start_epoch,checkpoint_dir=checkpoint_path,dataloaders=dataloaders)

  filename = "./model/"+"trial"+str(trialNumber)+".pth"
  torch.save(model.state_dict(), filename)


# %%
