#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
# %%
img = cv2.imread('/home/aicenter/Projects/demosaicing/test2.jpg')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# %%
fig = plt.figure(figsize=(15,15))
plt.imshow(rgb)
# %%
def remosaic(img):
    Ny, Nx, Nc = img.shape
    B = np.zeros([2*Ny, 2*Nx])
    for i in range(1,Ny):
        for j in range(1,Nx):
            B[2*i-1,2*j-1] = img[i,j,0]
            # B[2*i,2*j] = img[i,j,2]
            # B[2*i,2*j-1] = img[i,j,1]
            # B[2*i-1,2*j] = img[i,j,1]

    return B
#%%
img_ = remosaic(rgb[:100,:100])
            

# %%
fig = plt.figure(figsize=(15,15))
plt.imshow(img_)

fig = plt.figure(figsize=(15,15))
plt.imshow(img[:100,:100,0])
# %%
input_shape ->  