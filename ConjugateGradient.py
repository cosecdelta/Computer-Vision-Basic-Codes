## Default modules imported. Import more if you need to.

import numpy as np
from scipy.signal import convolve2d as conv2
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
# Implement using conjugate gradient, with a weight = 0 for mask == 0, and proportional
# to n_z^2 elsewhere. See slides.

def ntod(nrm, mask, lmda):
    weight = np.square(nrm[:,:,2])
    weight = weight*mask                              #Multiplying the weights with the mask
    gx = -np.divide(nrm[:,:,0],nrm[:,:,2])            #Generating the gradients in x and y directions
    gy = -np.divide(nrm[:,:,0],nrm[:,:,2])
    gx[np.isnan(gx)] = 0
    gy[np.isnan(gy)] = 0

    fx = np.reshape(np.array([0.5,0,-0.5]),(1,3))       #Generating the convolution filter
    fx_flip = np.flip(fx)                               #Flipping the fx filter
    fy = -np.transpose(fx)                              
    fy_flip = np.flip(fy)                               #Flipping the fy filter
    fr = np.array([[-1/9,-1/9,-1/9],[-1/9,8/9,-1/9],[-1/9,-1/9,-1/9]])      
    fr_flip = fr                             #Flipping the fr filter
    z = np.zeros(weight.shape)
    b1 = conv2(weight*gx, fx_flip, mode='same')         #Generating the b vector
    b2 = conv2(weight*gy, fy_flip, mode='same')
    b = b1 + b2
    r = b                                               #The QZ expression is zero as Z is a zero matrix initially so r=b
    p = np.copy(r)
    iterations = 100
    for i in range(iterations):
        q1 = conv2(conv2(p,fx,mode='same')*weight,fx_flip,mode='same')
        q2 = conv2(conv2(p,fy,mode='same')*weight,fy_flip, mode='same')
        q3 = conv2(conv2(p,fr,mode='same'),fr_flip, mode='same')*lmda
        qp = q1 + q2 +q3                                                       #Generating the QP term 
        
        alpha = np.inner(np.reshape(r,(-1)),np.reshape(r,(-1)))/np.inner(np.reshape(p,(-1)),np.reshape(qp,(-1)))       #Generating and updating the Alpha term 
        z = z + np.multiply(alpha,p)                                                                                    #Generating and updating the Z term i.e. the depth map
        r_old = np.copy(r)
        r = r_old - np.multiply(alpha,qp)                                                                                   #Generating and updating the r term 
        beta = np.inner(np.reshape(r,(-1)),np.reshape(r,(-1)))/np.inner(np.reshape(r_old,(-1)),np.reshape(r_old,(-1)))  #Generating and updating the Beta term 
        p = r + np.multiply(beta,p)                                                                                      #Generating and updating the p term 

    return z                                          #final depth map


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
Z = ntod(nrm,mask,1e-7)


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
