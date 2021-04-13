import numpy as np

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

from model import UNet
from torch.utils.data import Dataset, DataLoader

from dataset import ToBayer
from utils import *

from tqdm import tqdm
from collections import defaultdict
import os

import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class McMaster_Dataset(Dataset):
    def __init__(self, txt_path, img_dir,transform):
        """
        Initialize data set as a list of IDs corresponding to each item of data set

        :param img_dir: path to image files as a uncompressed tar archive
        :param txt_path: a text file containing names of all of images line by line
        :param transform: apply some transforms like cropping, rotating, etc on input image
        """

        self.df = pd.read_csv(txt_path, delim_whitespace=True,header=None)
        self.img_names = self.df.index.values
        self.txt_path = txt_path
        self.img_dir = img_dir
        self.transform = transform

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
        # left = width//2-16
        # top = height//2+16
        # right = left + 31
        # bottom = top -31
        left = 0
        top = 0
        right = 32
        bottom = 32
        # img.crop((left, top, right, bottom))

        img1 = img.crop((left, top, right, bottom))

        img1 = self.transform(img1)
        # img.astype(np.float32)
        # print("img size:",img.shape)
        return img1.astype(np.float32),img1.astype(np.float32)
        # return img

def print_metrics(metrics, epoch_samples, phase):    
      outputs = []
      for k in metrics.keys():
          outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    
      print("\n {}: {}".format(phase, ", ".join(outputs)))  

def test_model(model,dataloader):
  with torch.no_grad():
    model.eval()
    metrics = defaultdict(float)
    epoch_samples = 0
    for inputs,targets in tqdm(dataloader):
      inputs = inputs.to(device)
      targets = targets.to(device)
      # print(inputs.shape)
      batch_size, Nc, Ny, Nx = inputs.shape
      mask_original, mask_interm = create_masks(batch_size, Nc, Ny, Nx)
      mask_interm = mask_interm.to(device)
      mask_original = mask_original.to(device)

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

      # statistics
      epoch_samples += inputs.size(0)

    print_metrics(metrics, epoch_samples,'test')

def get_mcmaster_loader():
    txt_path = "./mcmaster_path.csv"
    img_dir = "./data/McM"
    mcmaster_dataset = McMaster_Dataset(txt_path=txt_path, img_dir=img_dir, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(mcmaster_dataset, batch_size=1, 
                                                shuffle=True, num_workers=8)
    return test_dataloader

if __name__ == "__main__":
  dataloader = get_mcmaster_loader()
  model_path_retrieve = "./model/trial1.pth"
  model = UNet(n_class=3)
  model.load_state_dict(torch.load(model_path_retrieve))
  model = model.to(device)
  test_model(model, dataloader)
  # if not os.path.exists('model'):
  #   os.makedirs('model')
  # filename = "./model/"+"trial"+str(trialNumber)+".pth"
  # torch.save(model.state_dict(), filename)