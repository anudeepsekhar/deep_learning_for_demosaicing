import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import copy
import time
import pandas as pd
from PIL import Image
from torchvision.transforms import ToPILImage 
from torch.utils.data import Dataset, DataLoader

from model import Resnet34,SimpleResidualBlock
from model_utils import *
from dataset import McMaster_Dataset,get_mcmaster_loader

from tqdm import tqdm
from collections import defaultdict
import os

import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trialNumber = 1

def test_model(model,dataloader,image_saving_dir):
  print("Testing...")
  with torch.no_grad():
    model.eval()
    test_loss = 0.0
    for batch,(inputs,targets) in enumerate(tqdm(dataloader)):
      inputs = inputs.to(device)
      targets = targets.to(device)

      batch_size, Nc, Ny, Nx = inputs.shape
      model_output = model(inputs)

      # pdb.set_trace()
      # replace with original correct pixel values in certain locations
      result = model_output.clone()
  
      # green channel
      for k in range(batch_size):
        for i in range(0,Ny,1): #row
          for j in range(0,Nx,2): #column
            if (i%2)==0: # even rows
              result[k,1,i,j+1] = targets[k,1,i,j+1]
            elif (i%2)==1: # odd rows
              result[k,1,i,j] = targets[k,1,i,j]

      # red channel and blue channel
      for i in range(0,Ny,2):
        for j in range(0,Nx,2):
          result[k,2,i,j] = targets[k,2,i,j] # blue channel
          result[k,0,i+1,j+1] = targets[k,0,i+1,j+1] # red channel

      # Calculate L2 loss
      criterion = nn.MSELoss()
      loss = criterion(result, targets)

      test_loss += loss.item()

      # compute peak snr
      # max pixel value
      peak_pixel_value = torch.max(result)
      # MSE
      loss = nn.MSELoss(reduction='mean')
      mse = loss(result,targets)

      peak_snr = 10*math.log10(peak_pixel_value**2/mse)
      print('peak_snr: {:.2f} decibel'.format(peak_snr))

      # saves the image for visual comparison
      file_name = 'McM_'+str(batch+1)+'.png'
      dir_path = image_saving_dir +'/'+file_name
      torchvision.utils.save_image(torch.squeeze(result),dir_path)
      # result_img = ToPILImage()(torch.squeeze(result)) # squeeze to get rid of the extra dimension of 1 batch
      # result_img.save(dir_path,"PNG")

    num_batch = len(dataloader)
    test_loss /= num_batch
    # print_metrics(metrics, num_batch, 'train')
    print('Average test loss: {:.6f}'.format(test_loss))


if __name__ == "__main__":
  dataloader = get_mcmaster_loader()
  model_path_retrieve = "./model/"+"trial"+str(trialNumber)+".pth"
  in_features = 3 # RGB channels
  model = Resnet34(SimpleResidualBlock,in_features)
  model.load_state_dict(torch.load(model_path_retrieve))
  model = model.to(device)

  if not os.path.exists('./data/McM_outs_resnet'):
    os.makedirs('./data/McM_outs_resnet')
  image_saving_dir = './data/McM_outs_resnet'
  test_model(model, dataloader,image_saving_dir)
