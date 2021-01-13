## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave
from scipy.signal import convolve2d as conv2


# Fill this out
def kernpad(K,size):
    kernel_size = np.array(K.shape)
    image_size = np.array(size)
    pad_length = image_size - kernel_size
    center_pixel = (kernel_size-1)//2
    padded_img = np.pad(K,((0,pad_length[0]),(0,pad_length[1])))
    circular_rotate_kernel = np.roll(padded_img, -int(center_pixel[0]), axis=0)              #Rotating along the rows
    circular_rotate_kernel = np.roll(padded_img, -int(center_pixel[1]), axis=1)              #Rotating along the columns
    
    # Placeholder code
    return circular_rotate_kernel

########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img = np.float32(imread(fn('inputs/p5_inp.jpg')))/255.

# Create Gaussian Kernel
x = np.float32(range(-21,22))
x,y = np.meshgrid(x,x)
G = np.exp(-(x*x+y*y)/2/9.)
G = G / np.sum(G[:])

# Traditional convolve
v1 = conv2(img,G,'same','wrap')

# Convolution in Fourier domain
G = kernpad(G,img.shape)
v2f = np.fft.fft2(G)*np.fft.fft2(img)
v2 = np.real(np.fft.ifft2(v2f))

# Stack them together and save
out = np.concatenate([img,v1,v2],axis=1)
out = np.minimum(1.,np.maximum(0.,out))

imsave(fn('outputs/prob5.jpg'),out)


                 
