## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave

# Fill this out
# X is input color image
# K is the support of the filter (2K+1)x(2K+1)
# sgm_s is std of spatial gaussian
# sgm_i is std of intensity gaussian
def bfilt(X,K,sgm_s,sgm_i):
    
    # Placeholder
    Y = np.zeros(X.shape)
    length = X.shape[0]
    breadth = X.shape[1]
    depth = X.shape[2]
    filt_sum = np.zeros([length,breadth,depth])
    for x in range(-K,K):
        for y in range(-K,K):
            if x>0:
                x_start = x
                x_end = length   
            else:
                x_start = 0
                x_end = length - x
                
            if y>0:
                y_start = y
                y_end = breadth     
            else:
                y_start = 0
                y_end = breadth - y
                
            X_dash = np.roll(X,x,axis=0)
            X_dash_new = np.roll(X_dash,y,axis=1)
            X_diff = (X[x_start:x_end,y_start:y_end,:] - X_dash_new[x_start:x_end,y_start:y_end,:])**2
            filt = np.exp(-(x**2 + y**2)/(2*(sgm_s**2))-X_diff/(2*(sgm_i**2)))
            Y[x_start:x_end,y_start:y_end,:] += filt*X_dash_new[x_start:x_end,y_start:y_end,:] 
            filt_sum[x_start:x_end,y_start:y_end,:] +=filt
    Y = Y/filt_sum
    
    return Y


########################## Support code below

def clip(im):
    return np.maximum(0.,np.minimum(1.,im))
    

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img1 = np.float32(imread(fn('inputs/p4_nz1.jpg')))/255.
img2 = np.float32(imread(fn('inputs/p4_nz2.jpg')))/255.

K=9
print("Creating outputs/prob4_1_a.jpg")
im1A = bfilt(img1,K,2,0.5)
imsave(fn('outputs/prob4_1_a.jpg'),clip(im1A))


print("Creating outputs/prob4_1_b.jpg")
im1B = bfilt(img1,K,4,0.25)
imsave(fn('outputs/prob4_1_b.jpg'),clip(im1B))

print("Creating outputs/prob4_1_c.jpg")
im1C = bfilt(img1,K,16,0.125)
imsave(fn('outputs/prob4_1_c.jpg'),clip(im1C))

#Repeated application
print("Creating outputs/prob4_1_rep.jpg")
im1D = bfilt(img1,K,2,0.125)
for i in range(8):
    im1D = bfilt(im1D,K,2,0.125)
imsave(fn('outputs/prob4_1_rep.jpg'),clip(im1D))

# Try this on image with more noise    
print("Creating outputs/prob4_2_rep.jpg")
im2D = bfilt(img2,K,8,0.125)
for i in range(16):
    im2D = bfilt(im2D,K,2,0.125)
imsave(fn('outputs/prob4_2_rep.jpg'),clip(im2D))
