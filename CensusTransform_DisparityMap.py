## Default modules imported. Import more if you need to.

import numpy as np


#########################################
### Hamming distance computation
### You can call the function hamdist with two
### uint32 bit arrays of the same size. It will
### return another array of the same size with
### the elmenet-wise hamming distance.
hd8bit = np.zeros((256,))
for i in range(256):
    v = i
    for k in range(8):
        hd8bit[i] = hd8bit[i] + v%2
        v=v//2

def hamdist(x,y):
    dist = np.zeros(x.shape)
    g = x^y
    for i in range(4):
        dist = dist + hd8bit[g%256]
        g = g // 256
    return dist
#########################################


## Fill out these functions yourself
# Compute a 5x5 census transform of the grayscale image img.
# Return a uint32 array of the same shape
def census(img):
    W = img.shape[1]
    H = img.shape[0]
    disparity_pixels = np.ndarray((H,W))
    for i in range(H):
      for j in range(W):
          count = 0
          disparity = 0
          for k in range(i-2,i+3):
            for l in range(j-2,j+3):
                if(k<0 or l<0 or k>H-1 or l>W-1):
                    disparity = disparity + (2**count)*0
                    count = count + 1
                elif(k==i and l==j):
                    disparity = disparity + 0
                elif img[i,j]>img[k,l]:
                    disparity = disparity + (2**count)*1
                    count = count + 1
                elif img[i,j]<=img[k,l]:
                    disparity = disparity + (2**count)*0                   
                    count = count + 1
            
          disparity_pixels[i,j] = disparity     
    return disparity_pixels
    

# Given left and right image and max disparity D_max, return a disparity map
# based on matching with  hamming distance of census codes. Use the census function
# you wrote above.
#
# d[x,y] implies that left[x,y] matched best with right[x-d[x,y],y]. Disparity values
# should be between 0 and D_max (both inclusive).


def smatch(left,right,dmax):
    
    H = left.shape[0]
    W = left.shape[1]
    d_map = np.zeros((H,W))
    #test_array = 24 * np.ones([left.shape[0],left.shape[1],dmax+1], dtype=np.float32)
    census_right = np.uint8(census(right))
    census_left = np.uint8(census(left))
    for i in range(W-1,0,-1):
        left_image = np.reshape(census_left[:,i],(H,1))                 #Taking each column of the left image(say D) and comparing to (D-40) columns of the right image
        right_image = census_right[:,i:np.maximum(i-dmax,0):-1]
        hamming_map = hamdist(left_image,right_image)
        #test_array[:,i,:] = hamming_map
        d_map[:,i] = np.argmin(hamming_map, axis=1)
        
    return d_map, census_left, census_right
    
    
########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')


left = imread(fn('inputs/left.jpg'))
right = imread(fn('inputs/right.jpg'))

dmax = 40
H = left.shape[0]
W = left.shape[1]
d_map = np.zeros((H,W))
test_array = 24 * np.ones([left.shape[0],left.shape[1],dmax+1], dtype=np.float32)
census_right = np.uint8(census(right))
census_left = np.uint8(census(left))
for i in range(W-1,0,-1):
    left_image = np.reshape(census_left[:,i],(H,1))                 #Taking each column of the left image(say D) and comparing to (D-40) columns of the right image
    right_image = census_right[:,i:np.maximum(i-dmax,0):-1]
    hamming_map = hamdist(left_image,right_image)
    
    d_map[:,i] = np.argmin(hamming_map, axis=1)


d = d_map
#d, census_l, census_r = smatch(left,right,40)

# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d.flatten())/20.))[:,0:3]
dimg = dimg.reshape([d.shape[0],d.shape[1],3])
imsave(fn('outputs/prob5.png'),dimg)
# imsave(fn('outputs/prob5l.png'),census_l)
# imsave(fn('outputs/prob5r.png'),census_r)
