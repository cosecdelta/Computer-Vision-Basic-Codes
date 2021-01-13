## Default modules imported. Import more if you need to.
import numpy as np
from scipy.signal import convolve2d as conv2

# Use these as the x and y derivative filters
fx = np.float32([[1,0,-1]]) * np.float32([[1,1,1]]).T / 6.
fy = fx.T


# Compute optical flow using the lucas kanade method
# Use the fx, fy, defined above as the derivative filters
# and compute derivatives on the average of the two frames.
# Also, consider (x',y') values in a WxW window.
# Return two image shape arrays u,v corresponding to the
# horizontal and vertical flow.

def lucaskanade(f1,f2,W):
    I_avg = (f1+f2)/2
    I_t = f1-f2
    I_x = conv2(I_avg,fx,boundary='symm', mode='same')
    I_y = conv2(I_avg,fy,boundary='symm', mode='same')
    w = np.ones((W,W))
    epsilon = 1e-8
    I_xy = conv2(np.multiply(I_x,I_y),w,boundary='symm', mode='same')
    I_x2 = conv2(np.square(I_x),w,boundary='symm', mode='same') + epsilon
    I_y2 = conv2(np.square(I_y),w,boundary='symm', mode='same') + epsilon
    I_xt = conv2(np.multiply(I_x,I_t),w,boundary='symm', mode='same')
    I_yt = conv2(np.multiply(I_y,I_t),w,boundary='symm', mode='same')

    u = (np.multiply(I_xy,I_yt) - np.multiply(I_y2,I_xt))/np.multiply(I_x2,I_y2) - np.square(I_xy)
    v = (np.multiply(I_xy,I_xt) - np.multiply(I_x2,I_yt))/np.multiply(I_x2,I_y2) - np.square(I_xy)
   
    return u,v

########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


f1 = np.float32(imread(fn('inputs/frame10.jpg')))/255.
f2 = np.float32(imread(fn('inputs/frame11.jpg')))/255.

u,v = lucaskanade(f1,f2,11)


# Display quiver plot by downsampling
x = np.arange(u.shape[1])
y = np.arange(u.shape[0])
x,y = np.meshgrid(x,y[::-1])
plt.quiver(x[::8,::8],y[::8,::8],u[::8,::8],-v[::8,::8],pivot='mid')

plt.show()
