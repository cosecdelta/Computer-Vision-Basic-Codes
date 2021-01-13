## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave


## Fill out these functions yourself


# Inputs:
#    nrm: HxWx3. Unit normal vectors at each location. All zeros at mask == 0
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#    lmda: Scalar value of lambda to be used for regularizer weight as in slides.
#
# Returns depth map Z of size HxW.
#
# Be careful about division by 0.
#
# Implement in Fourier Domain / Frankot-Chellappa
def kernpad(K,size):
    kernel_size = np.array(K.shape)
    image_size = np.array(size)
    pad_length = image_size - kernel_size
    center_pixel = (kernel_size-1)//2
    padded_img = np.pad(K,((0,pad_length[0]),(0,pad_length[1])))
    circular_rotate_kernel = np.roll(padded_img, -int(center_pixel[0]), axis=0)              #Rotating along the rows
    circular_rotate_kernel = np.roll(padded_img, -int(center_pixel[1]), axis=1)              #Rotating along the columns
    return circular_rotate_kernel

def ntod(nrm, mask, lmda):
    nrm_flat = np.reshape(nrm,(-1,3))
    mask_flat = np.broadcast_to(np.reshape(mask,(-1,1)),nrm_flat.shape)
    nrm_flat = nrm_flat*mask_flat
    nrm_flat = np.reshape(nrm_flat,nrm.shape)
    gx = -np.divide(nrm_flat[:,:,0],nrm_flat[:,:,2])
    gy = -np.divide(nrm_flat[:,:,0],nrm_flat[:,:,2])
    gx[np.isnan(gx)] = 0
    gy[np.isnan(gy)] = 0
    gu = np.fft.fft2(gx)
    gv = np.fft.fft2(gy)
    fx = np.reshape(np.array([0.5,0,-0.5]),(1,3))
    fy = -np.transpose(fx)
    fu = np.fft.fft2(kernpad(fx, gu.shape))
    fv = np.fft.fft2(kernpad(fy, gv.shape))
    fr = np.array([[-1/9,-1/9,-1/9],[-1/9,8/9,-1/9],[-1/9,-1/9,-1/9]])
    fr_uv = np.fft.fft2(kernpad(fr, gu.shape))
    
    fz_final = (np.conj(fu)*gu + np.conj(fv)*gv)/(np.square(abs(fu)) + np.square(abs(fv)) + lmda*np.square(abs(fr_uv)))
    final_depth = np.real(np.fft.ifft2(fz_final))
    
    return final_depth


########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

#### Main function

#nrm = imread(fn('inputs/phstereo/true_normals.png'))
# Un-comment  next line to read your output instead
nrm = imread(fn('outputs/prob3_nrm.png'))

mask = np.float32(imread(fn('inputs/phstereo/mask.png')) > 0)

nrm = np.float32(nrm/255.0)
nrm = nrm*2.0-1.0
nrm = nrm * mask[:,:,np.newaxis]


# Main Call
Z = ntod(nrm,mask,1e-6)


# Plot 3D shape

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x,y = np.meshgrid(np.float32(range(nrm.shape[1])),np.float32(range(nrm.shape[0])))
x = x - np.mean(x[:])
y = y - np.mean(y[:])

Zmsk = Z.copy()
Zmsk[mask == 0] = np.nan
Zmsk = Zmsk - np.nanmedian(Zmsk[:])

lim = 100
ax.plot_surface(x,-y,Zmsk, \
                linewidth=0,cmap=cm.inferno,shade=True,\
                vmin=-lim,vmax=lim)

ax.set_xlim3d(-450,450)
ax.set_ylim3d(-450,450)
ax.set_zlim3d(-450,450)

plt.show()
