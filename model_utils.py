import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

def save_ckp(state, checkpoint_dir):
    torch.save(state, checkpoint_dir)

# create masks where true pixel locations have value zero
def create_RGB_masks(img):
  Nc, Ny, Nx = img.shape
  R = np.ones([Ny, Nx])
  G = np.ones([Ny, Nx])
  B = np.ones([Ny, Nx])

  # green channel
  for i in range(0,Ny,1): #row
    for j in range(0,Nx,2): #column
      if (i%2)==0: # even rows
        G[i,j+1] = 0
      elif (i%2)==1: # odd rows
        G[i,j] = 0

  # red channel and blue channel
  for i in range(0,Ny,2):
    for j in range(0,Nx,2):
      B[i,j] = 0 # blue channel
      R[i+1,j+1] = 0 # red channel
  
  return torch.from_numpy(np.array([R,G,B]))

def calc_loss(outputs, labels, metrics):
    criterion = nn.MSELoss()
    loss = criterion(outputs, labels)

    # metrics['loss'] += loss.data.cpu().numpy() * labels.shape[0]
    metrics['loss'] += loss.item()
    return loss


def replace_with_original(img_original, model_output):
  print("model_output.shape: ",model_output.shape)
  batch, Nc, Ny, Nx = model_output.shape
  result = model_output.clone()
  # green channel
  for k in range(batch):
    for i in range(0,Ny,1): #row
      for j in range(0,Nx,2): #column
        if (i%2)==0: # even rows
          result[k,1,i,j+1] = img_original[k,1,i,j+1]
        elif (j%2)==1: # odd rows
          result[k,1,i,j] = img[k,1,i,j]

    # red channel and blue channel
    for i in range(0,Ny,2):
      for j in range(0,Nx,2):
        result[k,2,i,j] = img[k,2,i,j] # blue channel
        result[k,0,i+1,j+1] = img[k,0,i+1,j+1] # red channel

  return result

def print_metrics(metrics, num_batch, phase):    
      outputs = []
      for k in metrics.keys():
          outputs.append("{}: {:4f}".format(k, metrics[k] / num_batch))
    
      print("\n {}: {}".format(phase, ", ".join(outputs)))  

  