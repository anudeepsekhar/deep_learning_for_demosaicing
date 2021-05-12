#%%
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import copy
import time
import pdb

from dataset import get_CUB200_loader
from model import Net_Superresolution,REDNet_model,SRCNN,model_with_upsampling,VDSR_Net
from training import train_model

from tqdm import tqdm
import os
import math
from PIL import Image

import pickle
import argparse

# resume_from_ckp = False
# trialNumber = 13
# checkpoint_path = "./checkpoint_SRCNN/"+"trial"+str(trialNumber)+"checkpoint.pt"
# num_epochs = 3

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def lr_checker(lr):
    lr = float(lr)
    if lr < 1e-6 or lr > 1e-1:
        raise argparse.ArgumentTypeError('invalid learning rate, it must be between 1e-6 and 1e-1')
    return lr

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#%%
if __name__ == "__main__":
  # Create the parser
  parser = argparse.ArgumentParser()

  # Add an argument
  parser.add_argument("--resume_from_ckp", type=str2bool, nargs='?',
                        default=False,
                        help="Resume from checkpoint.")
  parser.add_argument("--trial_number", type=int, required=True,
                        help="Trial number.")
  parser.add_argument("--model", type=str, required=True,
                        help="The model you want to use.")
  parser.add_argument("--num_epochs", type=int, required=True,
                        help="Number of epochs.")
  parser.add_argument("--lr", type = lr_checker, default = 1e-4,
                        help="Learning rate.")
  parser.add_argument("--not_cropped", type=str2bool, nargs='?',
                        default=False,
                        help="Images will not be cropped.")

  # Parse the argument
  args = parser.parse_args()

  # assign the variables according to the argument values
  resume_from_ckp = args.resume_from_ckp
  trialNumber = args.trial_number
  num_epochs = args.num_epochs
  lr = args.lr

  if args.model == 'deep_residual_network':
    model = Net_Superresolution(withRedNet=False,withSRCNN=False)
    dataloaders = get_CUB200_loader()
    if not os.path.exists('checkpoint_superresolution'):
      os.makedirs('checkpoint_superresolution')
    checkpoint_path = "./checkpoint_superresolution/"+"trial"+str(trialNumber)+"checkpoint.pt"
  elif args.model == 'deep_residual_network_rednet':
    model = Net_Superresolution(withRedNet=True,withSRCNN=False)
    dataloaders = get_CUB200_loader()
    if not os.path.exists('checkpoint_superresolution'):
      os.makedirs('checkpoint_superresolution')
    checkpoint_path = "./checkpoint_superresolution/"+"trial"+str(trialNumber)+"checkpoint.pt"
  elif args.model == 'deep_residual_network_SRCNN':
    model = Net_Superresolution(withRedNet=False,withSRCNN=True)
    dataloaders = get_CUB200_loader()
    if not os.path.exists('checkpoint_superresolution'):
      os.makedirs('checkpoint_superresolution')
    checkpoint_path = "./checkpoint_superresolution/"+"trial"+str(trialNumber)+"checkpoint.pt"
  elif args.model == 'rednet':
    model = REDNet_model()
    if not os.path.exists('checkpoint_rednet'):
      os.makedirs('checkpoint_rednet')
    checkpoint_path = "./checkpoint_rednet/"+"trial"+str(trialNumber)+"checkpoint.pt"
    dataloaders = get_CUB200_loader(three_channel_bayer=True)
  elif args.model == 'SRCNN':
    SRCNN_model = SRCNN(num_channels=3)
    model = model_with_upsampling(SRCNN_model)
    if not os.path.exists('checkpoint_SRCNN'):
      os.makedirs('checkpoint_SRCNN')
    checkpoint_path = "./checkpoint_SRCNN/"+"trial"+str(trialNumber)+"checkpoint.pt"
    if args.not_cropped:
      dataloaders = get_CUB200_loader(three_channel_bayer=True,no_crop=True,batch_size=8)
    else:
      dataloaders = get_CUB200_loader(three_channel_bayer=True)
  
  elif args.model == 'VDSR':
    VDSR_model = VDSR_Net()
    model = model_with_upsampling(VDSR_model)
    if not os.path.exists('checkpoint_VDSR'):
      os.makedirs('checkpoint_VDSR')
    checkpoint_path = "./checkpoint_VDSR/"+"trial"+str(trialNumber)+"checkpoint.pt"
    dataloaders = get_CUB200_loader(three_channel_bayer=True,no_crop=True,batch_size=2)
    # transform =  torchvision.transforms.Compose([
    #         torchvision.transforms.RandomApply(torch.nn.ModuleList([torchvision.transforms.RandomRotation(degrees=20)]),p=0.5),
    #         torchvision.transforms.RandomHorizontalFlip(p=0.5), # Flip the data horizontally
    #         torchvision.transforms.ToTensor()])
    # dataloaders = get_CUB200_loader(three_channel_bayer=True,no_crop=True,batch_size=8,transform=transform)
  else:
    raise argparse.ArgumentError("Invalid model")


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

  model = model.to(device)
  if args.model == 'VDSR':
    weightDecay = 1e-4
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,weight_decay=weightDecay,momentum=0.9) # lr should be set to 0.1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.1,verbose=True)
  else:
    weightDecay = 5e-5
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
  

  if args.model == 'VDSR':
    model = train_model(model, optimizer, scheduler, num_epochs=num_epochs, start_epoch=start_epoch,checkpoint_dir=checkpoint_path,dataloaders=dataloaders,clip_norm=0.4,scheduler_no_arg=True)
  else:
    model = train_model(model, optimizer, scheduler, num_epochs=num_epochs, start_epoch=start_epoch,checkpoint_dir=checkpoint_path,dataloaders=dataloaders)

  filename = "./model/"+"trial"+str(trialNumber)+".pth"
  torch.save(model.state_dict(), filename)


# %%
