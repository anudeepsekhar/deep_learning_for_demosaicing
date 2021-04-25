import numpy as np
import pandas as pd
from PIL import Image
import os
import os.path
from dataset import To_Bayer,DownSize,preprocess_CUB200_Dataset

if __name__ == "__main__":
  save_path = "./data/CUB200_processed"
  txt_file_path = "./data/CUB_200_2011/images.txt"
  img_dir = "./data/CUB_200_2011/images"
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  preprocess_CUB200_Dataset(save_path, txt_file_path,img_dir)