import numpy as np
import pandas as pd
from PIL import Image
import os
import os.path
from dataset import To_Bayer,preprocess_CUB200_Dataset,preprocess_CUB200_Dataset_no_crop
from model_utils import str2bool
import argparse

if __name__ == "__main__":
  # Create the parser
  parser = argparse.ArgumentParser()

  # Add an argument
  parser.add_argument("--crop", type=str2bool, nargs='?',
                        default=False,
                        help="Data preprocessing with cropping")

  # Parse the argument
  args = parser.parse_args()

  # assign the variables according to the argument values
  if args.crop:
    save_path = "./data/CUB200_processed"
  else:
    save_path = "./data/CUB200_processed_not_cropped"

  txt_file_path = "./data/CUB_200_2011/images.txt"
  img_dir = "./data/CUB_200_2011/images"
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  if args.crop:
    preprocess_CUB200_Dataset(save_path, txt_file_path,img_dir)
  else:
    preprocess_CUB200_Dataset_no_crop(save_path, txt_file_path,img_dir)