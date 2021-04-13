import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np


class Utils(Object):
  @staticmethod
  def calc_loss(outputs, labels, metrics):
      criterion = nn.MSELoss()
      loss = criterion(outputs, labels)

      metrics['loss'] += loss.data.cpu().numpy() * labels.shape[0]
      return loss

  @staticmethod
  def create_masks(batch_size, Nc, Ny, Nx):
          
          mask_original= np.zeros([batch_size, Nc, Ny, Nx])
          for b in range(batch_size):
              R = np.zeros([Ny, Nx])
              G = np.zeros([Ny, Nx])
              B = np.zeros([Ny, Nx]) 
              for i in range(1,Ny//2):
                  for j in range(1,Nx//2):
                      R[2*i-1,2*j-1] = True
                      B[2*i,2*j] = True
                      G[2*i,2*j-1] = True
                      G[2*i-1,2*j] = True
              mask_original[b] = np.array([R,G,B])



          # mask_interm = np.bitwise_not(mask_original)
          mask_interm = np.invert(mask_original.astype(np.bool))
                  
          return torch.Tensor(mask_original.astype(np.float32)), torch.Tensor(mask_interm.astype(np.float32))

  # @staticmethod
  # def print_metrics(metrics, epoch_samples, phase):    
  #     outputs = []
  #     for k in metrics.keys():
  #         outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    
  #     print("\n {}: {}".format(phase, ", ".join(outputs)))  


# def print_metrics(metrics, epoch_samples):    
#     outputs = []
#     for k in metrics.keys():
#         outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
#     print("\n test: {}".format(", ".join(outputs)))  

# transform = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         ToBayer()

#      ])