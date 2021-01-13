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
def census(img):
    W = img.shape[1]
    H = img.shape[0]
    c = np.zeros([H,W], dtype=np.uint32)
    
    inc = np.uint32(1)
    for dx in range(-2,3):
        for dy in range(-2,3): 
            if dx==0 and dy==0:
                continue
            cx0 = np.maximum(0,-dx); dx0 = np.maximum(0,dx)
            cx1 = W-dx0; dx1 = W-cx0;
            cy0 = np.maximum(0,-dy); dy0 = np.maximum(0,dy)
            cy1 = H-dy0; dy1 = H-cy0
            
            c[cy0:cy1,cx0:cx1] = c[cy0:cy1,cx0:cx1] + \
                inc*(img[cy0:cy1,cx0:cx1] > img[dy0:dy1,dx0:dx1])
            inc = inc*2
    
    return c

# Copy this from problem 2 solution.
def buildcv(left,right,dmax):

    cv = 24 * np.ones([left.shape[0],left.shape[1],dmax], dtype=np.float32)
    test = 24 * np.ones([left.shape[0],dmax], dtype=np.float32)
    census_right = census(right_g)
    census_left = census(left_g)
    H = left.shape[0]
    W = left.shape[1]
    for i in range(W-1,0,-1):
        left_image = np.reshape(census_left[:,i],(H,1))                
        right_image = census_right[:,i:np.maximum(i-dmax,0):-1]
        hamming_map = hamdist(left_image,right_image)
        if i-dmax<0:
            test = np.pad(hamming_map,[(0, 0), (0, abs(i-dmax))],mode='constant',constant_values=24)
        else:
            test = hamming_map 
        cv[:,i,:] = test
    return cv


# Do SGM. First compute the augmented / smoothed cost volumes along 4
# directions (LR, RL, UD, DU), and then compute the disparity map as
# the argmin of the sum of these cost volumes. 
def SGM(cv,P1,P2):
    H = cv.shape[0]
    W = cv.shape[1]
    D = cv.shape[2]
    d_ori = np.arange(D)
    d1 = np.broadcast_to(d_ori,[D,D])
    d2 = np.broadcast_to(d_ori,[D,D]).T
    S = abs(d1-d2)
    
    c_bar_lr = np.copy(cv)       
    for i in range(1,W):
        col_before = c_bar_lr[:,i-1,:]
        col_present = c_bar_lr[:,i,:]
        d_dash = np.repeat(col_before,D,axis=1)
        d_dash = np.reshape(d_dash,(H,D,D))
        final = d_dash + S
        final_cv = np.min(final,axis=1)
        final_z = np.argmin(final,axis=1)
        c_bar_lr[:,i,:] = col_present + final_cv 
                   
    c_bar_rl = np.copy(cv)
    for i in range(W-2,-1,-1):
        col_before = c_bar_rl[:,i+1,:]
        col_present = c_bar_rl[:,i,:]
        d_dash = np.repeat(col_before,D,axis=1)
        d_dash = np.reshape(d_dash,(H,D,D))
        final = d_dash + S
        final_cv = np.min(final,axis=1)
        final_z = np.argmin(final,axis=1)
        c_bar_rl[:,i,:] = col_present + final_cv 
        
    c_bar_ud = np.copy(cv)
    for i in range(1,H):
        col_before = c_bar_ud[i-1,:,:]
        col_present = c_bar_ud[i,:,:]
        d_dash = np.repeat(col_before,D,axis=1)
        d_dash = np.reshape(d_dash,(W,D,D))
        final = d_dash + S
        final_cv = np.min(final,axis=1)
        c_bar_ud[i,:,:] = col_present + final_cv 
            
    c_bar_du = np.copy(cv)
    for i in range(H-2,-1,-1):
        col_before = c_bar_du[i+1,:,:]
        col_present = c_bar_du[i,:,:]
        d_dash = np.repeat(col_before,D,axis=1)
        d_dash = np.reshape(d_dash,(W,D,D))
        final = d_dash + S
        final_cv = np.min(final,axis=1)
        c_bar_du[i,:,:] = col_present + final_cv 
    
    cv = np.argmin((c_bar_lr + c_bar_rl + c_bar_ud + c_bar_du),axis=2)
    return cv

    
    
########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')


left = np.float32(imread(fn('inputs/left.jpg')))/255.
right = np.float32(imread(fn('inputs/right.jpg')))/255.

left_g = np.mean(left,axis=2)
right_g = np.mean(right,axis=2)
                   
cv = buildcv(left_g,right_g,50)
d = SGM(cv,0.5,16)

# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d.flatten())/50.))[:,0:3]
dimg = dimg.reshape([d.shape[0],d.shape[1],3])
imsave(fn('outputs/prob3b.jpg'),dimg)
