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

from model import UNet
from model_utils import *
from dataset import *

from tqdm import tqdm
from collections import defaultdict
import os

import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trialNumber = 4
checkpoint_path = "./checkpoint/"+"trial"+str(trialNumber)+"checkpoint.pt"


# def print_metrics(metrics, epoch_samples, phase):    
#       outputs = []
#       for k in metrics.keys():
#           outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    
#       print("\n {}: {}".format(phase, ", ".join(outputs)))  

class McMaster_Dataset(Dataset):
    def __init__(self, txt_path, img_dir,transform,transform2):
        """
        Initialize data set as a list of IDs corresponding to each item of data set

        :param img_dir: path to image files as a uncompressed tar archive
        :param txt_path: a text file containing names of all of images line by line
        :param transform: apply some transforms like cropping, rotating, etc on input image
        :param transform2: applies toTensor() only
        """
        self.df = pd.read_csv(txt_path, delim_whitespace=True,header=None)
        self.img_names = self.df.index.values
        self.txt_path = txt_path
        self.img_dir = img_dir
        self.transform = transform
        self.transform2 = transform2

    def get_image_from_folder(self, name):
        """
        gets a image by a name gathered from file list text file

        :param name: name of targeted image
        :return: a PIL image
        """

        image = Image.open(os.path.join(self.img_dir, name))

        return image

    def __len__(self):
        """
        Return the length of data set using list of IDs

        :return: number of samples in data set
        """
        return len(self.img_names)

    def __getitem__(self, index):
        """
        Generate one item of data set.

        :param index: index of item in IDs list

        :return: a sample of data as a dict
        """

        img = self.get_image_from_folder(self.df.iloc[index,0])
        # crop into 32x32
        width, height = img.size
        left = width//2-16
        top = height//2-16
        right = left + 32
        bottom = top + 32
        # left = 0
        # top = 0
        # right = 32
        # bottom = 32
        # img.crop((left, top, right, bottom))

        img1_original = img.crop((left, top, right, bottom))

        img1 = self.transform(img1_original)

        img1_original = self.transform2(img1_original)

        return img1.float(),img1_original.float()


def get_mcmaster_loader():
    txt_path = "./mcmaster_path.csv"
    img_dir = "./data/McM"
    mcmaster_dataset = McMaster_Dataset(txt_path=txt_path, img_dir=img_dir, transform=transform, transform2=transform2)
    test_dataloader = torch.utils.data.DataLoader(mcmaster_dataset, batch_size=1, 
                                                shuffle=False, num_workers=8)
    return test_dataloader

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
    print('Average test loss: {:.2f}'.format(test_loss))







if __name__ == "__main__":
  dataloader = get_mcmaster_loader()
  model_path_retrieve = "./model/"+"trial"+str(trialNumber)+".pth"
  model = UNet(n_class=3)
  model.load_state_dict(torch.load(model_path_retrieve))
  model = model.to(device)

  if not os.path.exists('./data/McM_outs'):
    os.makedirs('./data/McM_outs')
  image_saving_dir = './data/McM_outs'
  test_model(model, dataloader,image_saving_dir)
  # if not os.path.exists('model'):
  #   os.makedirs('model')
  # filename = "./model/"+"trial"+str(trialNumber)+".pth"
  # torch.save(model.state_dict(), filename)