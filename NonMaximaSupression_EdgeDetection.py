## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave
from scipy.signal import convolve2d as conv2
from matplotlib import pyplot as plt

# Different thresholds to try
T0 = 0.5
T1 = 1.0
T2 = 1.5


########### Fill in the functions below ###############

dx = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])
dy = np.transpose(dx)

# Return magnitude, theta of gradients of X
def grads(X):
    #placeholder
    H = np.zeros(X.shape,dtype=np.float32)
    theta = np.zeros(X.shape,dtype=np.float32)
    grad_x = conv2(X,dx,boundary='symm', mode='same')
    grad_y = conv2(X,dy, boundary='symm', mode='same')
    H = np.sqrt(grad_x**2 + grad_y**2)
    theta = np.arctan2(grad_y,grad_x)

    return H,theta

def nms(E,H,theta):
    #placeholder
    theta_degrees = np.around(np.absolute((180/np.pi)*theta))
    theta_list = np.array([[0,45,90,135,180]])

    H = np.pad(H, pad_width = 1, mode='constant')
    E_new = np.copy(E)
    pos = np.array(np.where(E0==1))
    list_theta_values = theta_degrees[pos[0,:],pos[1,:]]
    list_theta_values_new = np.zeros(list_theta_values.size)
    count = 0
    for i in list_theta_values:
        temp = np.absolute(theta_list - i)
        temp_min_index = np.where(temp==np.min(temp))
        list_theta_values_new[count] = theta_list[temp_min_index]
        count = count + 1
    
    for i in range(pos.shape[1]):
        if list_theta_values_new[i] == 0:
            if (H[pos[0,i],pos[1,i]] > H[pos[0,i]-1,pos[1,i]] and H[pos[0,i],pos[1,i]] > H[pos[0,i]+1,pos[1,i]] ):
                E_new[pos[0,i],pos[1,i]] = 1
            else:
                E_new[pos[0,i],pos[1,i]] = 0
            
            if list_theta_values_new[i] == 45:
                if (H[pos[0,i],pos[1,i]] > H[pos[0,i]+1,pos[1,i]-1] and H[pos[0,i],pos[1,i]] > H[pos[0,i]+1,pos[1,i]-1] ):
                    E_new[pos[0,i],pos[1,i]] = 1
                else:
                    E_new[pos[0,i],pos[1,i]] = 0
            
            if list_theta_values_new[i] == 90:
                if (H[pos[0,i],pos[1,i]] > H[pos[0,i],pos[1,i]+1] and H[pos[0,i],pos[1,i]] > H[pos[0,i],pos[1,i]+1] ):
                    E_new[pos[0,i],pos[1,i]] = 1
                else:
                    E_new[pos[0,i],pos[1,i]] = 0
            
            if list_theta_values_new[i] == 135:
                if (H[pos[0,i],pos[1,i]] > H[pos[0,i]-1,pos[1,i]-1] and H[pos[0,i],pos[1,i]] > H[pos[0,i]+1,pos[1,i]+1] ):
                    E_new[pos[0,i],pos[1,i]] = 1
                else:
                    E_new[pos[0,i],pos[1,i]] = 0
            
            if list_theta_values_new[i] == 180:
                if (H[pos[0,i],pos[1,i]] > H[pos[0,i]-1,pos[1,i]] and H[pos[0,i],pos[1,i]] > H[pos[0,i]+1,pos[1,i]] ):
                    E_new[pos[0,i],pos[1,i]] = 1
                else:
                    E_new[pos[0,i],pos[1,i]] = 0

    return E_new

########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img = np.float32(imread(fn('inputs/p3_inp.jpg')))/255.

H,theta = grads(img)

imsave(fn('outputs/prob3_a.jpg'),H/np.max(H[:]))

## Part b

E0 = np.float32(H > T0)
E1 = np.float32(H > T1)
E2 = np.float32(H > T2)

imsave(fn('outputs/prob3_b_0.jpg'),E0)
imsave(fn('outputs/prob3_b_1.jpg'),E1)
imsave(fn('outputs/prob3_b_2.jpg'),E2)


E0n = nms(E0,H,theta)
E1n = nms(E1,H,theta)
E2n = nms(E2,H,theta)

imsave(fn('outputs/prob3_b_nms0.jpg'),E0n)
imsave(fn('outputs/prob3_b_nms1.jpg'),E1n)
imsave(fn('outputs/prob3_b_nms2.jpg'),E2n)
