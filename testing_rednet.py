import os
import math
from tqdm import tqdm
from PIL import Image
import pickle
import torch
from model import REDNet_model
import torch.nn as nn
import numpy as np
from testing_model import test_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trialNumber = 8

if __name__ == "__main__":
  var_save_dir = './data/variables'
  var_name = 'dataloaders'+'_trial'+str(trialNumber)+'.pkl'
  path = var_save_dir + '/' + var_name
  with open(path, 'rb') as file:
    # Call load method to deserialze
    dataloaders = pickle.load(file)

  model_path_retrieve = "./model/"+"trial"+str(trialNumber)+".pth"
  model = REDNet_model()
  model.load_state_dict(torch.load(model_path_retrieve))
  model = model.to(device)

  image_saving_dir = './data/CUB200_outs'
  if not os.path.exists(image_saving_dir):
    os.makedirs(image_saving_dir)
  test_model(model,image_saving_dir,dataloaders)