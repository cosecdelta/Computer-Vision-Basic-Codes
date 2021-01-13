### Default modules imported. Import more if you need to.
### DO NOT USE linalg.lstsq from numpy or scipy

import numpy as np
from skimage.io import imread, imsave
from scipy import linalg as lin

## Fill out these functions yourself


# Inputs:
#    imgs: A list of N color images, each of which is HxWx3
#    L:    An Nx3 matrix where each row corresponds to light vector
#          for corresponding image.
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#
# Returns nrm:
#    nrm: HxWx3 Unit normal vector at each location.
#
# Be careful about division by zero at mask==0 for normalizing unit vectors.
def pstereo_n(imgs, L, mask):
    img_flat = []
    for image in imgs:
        gray_scale = np.mean(image,axis=2)
        image_flat = np.reshape(gray_scale,(-1))*np.reshape(mask,(-1))
        img_flat.append(image_flat)
    
    img_flat = np.array(img_flat)
    a = np.matmul(np.transpose(L),L)
    b = np.matmul(np.transpose(L),img_flat)
    n_hat = lin.solve(a,b)                                          ## Solving by Cholesky Method
    n_norm = lin.norm(n_hat, axis=0)                                ## Normalizing the surface normals
    n_norm = np.broadcast_to(n_norm,(n_hat.shape))
    n_final = np.divide(n_hat,n_norm)
    n_final[np.isnan(n_final)] = 0
    n_image = np.reshape(np.transpose(n_final),image.shape)
    return n_image



# Inputs:
#    imgs: A list of N color images, each of which is HxWx3
#    nrm:  HxWx3 Unit normal vector at each location (from pstereo_n)
#    L:    An Nx3 matrix where each row corresponds to light vector
#          for corresponding image.
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#
# Returns alb:
#    alb: HxWx3 RGB Color Albedo values
#
# Be careful about division by zero at mask==0.
def pstereo_alb(imgs, nrm, L, mask):
    img_flat = []
    for image in imgs:
        image_flat = np.reshape(image,(-1,3))
        mask_flat = np.broadcast_to(np.reshape(mask,(-1,1)),image_flat.shape)
        image_flat = image_flat*mask_flat
        img_flat.append(image_flat)

    img_flat = np.array(img_flat)
    img_flat = np.reshape(img_flat,(15,-1))
    a = np.matmul(np.transpose(L),L)
    b = np.matmul(np.transpose(L),img_flat)
    n_hat = lin.solve(a,b)
    n_norm = lin.norm(n_hat, axis=0)                        ## As said in the lecture the n_norm is basically the albedos
    n_image = np.reshape(np.transpose(n_norm),image.shape)
    return n_image

    
    
    
########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

### Light directions matrix
L = np.float32( \
                [[  4.82962877e-01,   2.58819044e-01,   8.36516321e-01],
                 [  2.50000030e-01,   2.58819044e-01,   9.33012664e-01],
                 [ -4.22219593e-08,   2.58819044e-01,   9.65925813e-01],
                 [ -2.50000000e-01,   2.58819044e-01,   9.33012664e-01],
                 [ -4.82962966e-01,   2.58819044e-01,   8.36516261e-01],
                 [ -5.00000060e-01,   0.00000000e+00,   8.66025388e-01],
                 [ -2.58819044e-01,   0.00000000e+00,   9.65925813e-01],
                 [ -4.37113883e-08,   0.00000000e+00,   1.00000000e+00],
                 [  2.58819073e-01,   0.00000000e+00,   9.65925813e-01],
                 [  4.99999970e-01,   0.00000000e+00,   8.66025448e-01],
                 [  4.82962877e-01,  -2.58819044e-01,   8.36516321e-01],
                 [  2.50000030e-01,  -2.58819044e-01,   9.33012664e-01],
                 [ -4.22219593e-08,  -2.58819044e-01,   9.65925813e-01],
                 [ -2.50000000e-01,  -2.58819044e-01,   9.33012664e-01],
                 [ -4.82962966e-01,  -2.58819044e-01,   8.36516261e-01]])


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))


############# Main Program


# Load image data
imgs = []
for i in range(L.shape[0]):
    imgs = imgs + [np.float32(imread(fn('inputs/phstereo/img%02d.png' % i)))/255.]

mask = np.float32(imread(fn('inputs/phstereo/mask.png')) > 0)
   

nrm = pstereo_n(imgs,L,mask)
nimg = nrm/2.0+0.5
nimg = clip(nimg * mask[:,:,np.newaxis])
imsave(fn('outputs/prob3_nrm.png'),nimg)


alb = pstereo_alb(imgs,nrm,L,mask)

alb = alb / np.max(alb[:])
alb = clip(alb * mask[:,:,np.newaxis])

imsave(fn('outputs/prob3_alb.png'),alb)
