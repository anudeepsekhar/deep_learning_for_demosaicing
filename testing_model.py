import os
import math
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model(model,image_saving_dir,dataloaders):
  print("Testing...")
  with torch.no_grad():
    model.eval()
    test_loss = 0.0
    psnr_average = 0.0
    image_id = 0
    for batch,(inputs,targets) in enumerate(tqdm(dataloaders['test'])):
      inputs = inputs.to(device)
      targets = targets.to(device)
      batch_size, Nc, Ny, Nx = inputs.shape

      result = model(inputs)

      # replace with original correct pixel values in certain locations
  
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

      # Calculate loss
      criterion = nn.MSELoss()
      loss = criterion(result.float(), targets.float())

      test_loss += loss.item()

      # compute peak snr
      # max pixel value
      peak_pixel_value = torch.max(result)
      # MSE
      loss = nn.MSELoss(reduction='mean')
      mse = loss(result,targets)

      peak_snr = 10*math.log10(peak_pixel_value**2/mse)
      print('peak_snr: {:.2f} decibel'.format(peak_snr))
      psnr_average += peak_snr

      # pdb.set_trace()

      result = Image.fromarray((torch.squeeze(result).permute(1,2,0).cpu().detach().numpy()*255).astype(np.uint8))
      targets = Image.fromarray((torch.squeeze(targets).permute(1,2,0).cpu().detach().numpy()*255).astype(np.uint8))

      # saves the images and image paths to one same folder
      image_save_folder = image_saving_dir
      if not os.path.exists(image_save_folder):
        os.makedirs(image_save_folder)

      data_str = image_save_folder + '/' + str(image_id) +'_'+'result.TIF'
      label_str = image_save_folder + '/' + str(image_id) +'_'+ 'original.TIF'

      with open(image_saving_dir + '/test_result_paths.txt', 'a+') as txt:
        txt.write(data_str + ' ' + label_str + '\n')
        
      # # saves the images and image paths 
      # image_save_folder = image_saving_dir + '/' + str(image_id)
      # if not os.path.exists(image_save_folder):
      #   os.makedirs(image_save_folder)

      # data_str = image_save_folder + '/' + str(image_id) +'_'+'result.TIF'
      # label_str = image_save_folder + '/' + str(image_id) +'_'+ 'original.TIF'

      # with open(image_saving_dir + '/test_result_paths.txt', 'a+') as txt:
      #   txt.write(data_str + ' ' + label_str + '\n')
      
      targets.save(label_str, 'TIFF')
      result.save(data_str, 'TIFF')
      
      # saves the peak snr result
      with open(image_saving_dir + '/test_result_psnr.txt', 'a+') as txt:
        txt.write(str(image_id) + ' ' + str(peak_snr))

      image_id = image_id+1

    num_batch = len(dataloaders['test'])
    test_loss /= num_batch
    psnr_average /= num_batch

    print('Average test loss: {:.6f}'.format(test_loss))
    print('Average psnr: {:.6f}'.format(psnr_average))