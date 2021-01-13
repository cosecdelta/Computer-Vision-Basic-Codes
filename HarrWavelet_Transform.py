## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave
from numpy.linalg import inv

## Fill out these functions yourself

def im2wv(img,nLev):
    # Placeholder that does nothing
    harr_kernel = np.multiply(np.array([[1,1,1,1],[-1,1,-1,1],[-1,-1,1,1],[1,-1,-1,1]]),0.5)
    img_new = img
    final = []
    for count in range(nLev):
        L,H1,H2,H3 = [],[],[],[]
        for i in range(0,img_new.shape[0],2):
            for j in range(0,img_new.shape[1],2):
                a = img_new[i,j]
                b = img_new[i+1,j]
                c = img_new[i,j+1]
                d = img_new[i+1,j+1]
                inp = np.array([[a],[b],[c],[d]])
                res = np.matmul(harr_kernel,inp)
                L.append(res[0])
                H1.append(res[1])
                H2.append(res[2])
                H3.append(res[3])
        
        L = np.array(L).reshape(int(img_new.shape[0]/2),int(img_new.shape[0]/2))
        H1 = np.array(H1).reshape(int(img_new.shape[0]/2),int(img_new.shape[0]/2))
        H2 = np.array(H2).reshape(int(img_new.shape[0]/2),int(img_new.shape[0]/2))
        H3 = np.array(H3).reshape(int(img_new.shape[0]/2),int(img_new.shape[0]/2))
        temp = [H1,H2,H3]
        final.append(temp)
        img_new = L
           
    final.append(L)
    
    return final


def wv2im(pyr):
    # Placeholder that does nothing
    harr_kernel = np.multiply(np.array([[1,1,1,1],[-1,1,-1,1],[-1,-1,1,1],[1,-1,-1,1]]),0.5)
    harr_kernel_inv = inv(harr_kernel)
    steps = len(pyr) - 1
    L_new = pyr[-1]
    for count in range(steps,0,-1):
        level = count - 1
        temp_list = pyr[level]
        H1 = temp_list[0]
        H2 = temp_list[1]
        H3 = temp_list[2]
        img_temp = []
        for i in range(0,L_new.shape[0]):
            for j in range(0,L_new.shape[1]):
                wave_coeff = np.array([[L_new[i,j],H1[i,j],H2[i,j],H3[i,j]]])
                res = np.matmul(harr_kernel_inv, wave_coeff.T)
                img_temp.append(res)
    
        img_temp = np.array(img_temp)
        img_new = np.zeros((int(L_new.shape[0]*2),int(L_new.shape[1]*2)))
        count = 0
        for i in range(0,L_new.shape[0]*2,2):
            for j in range(0,L_new.shape[0]*2,2):
                img_new[i,j] = img_temp[count,0,:]
                img_new[i+1,j] = img_temp[count,1,:]
                img_new[i,j+1] = img_temp[count,2,:]
                img_new[i+1,j+1] = img_temp[count,3,:]
                count = count + 1
        
        L_new = img_new    
    
    return L_new



########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))


# Visualize pyramid like in slides
def vis(pyr, lev=0):
    if len(pyr) == 1:
        return pyr[0]/(2**lev)

    sz=pyr[0][0].shape
    sz1 = [sz[0]*2,sz[1]*2]
    img = np.zeros(sz1,dtype=np.float32)

    img[0:sz[0],0:sz[1]] = vis(pyr[1:],lev+1)

    # Just scale / shift gradient images for visualization
    img[sz[0]:,0:sz[1]] = pyr[0][0]*(2**(1-lev))+0.5
    img[0:sz[0],sz[1]:] = pyr[0][1]*(2**(1-lev))+0.5
    img[sz[0]:,sz[1]:] = pyr[0][2]*(2**(1-lev))+0.5

    return img



############# Main Program


img = np.float32(imread(fn('inputs/p6_inp.jpg')))/255.

# Visualize pyramids
pyr = im2wv(img,1)
imsave(fn('outputs/prob6a_1.jpg'),clip(vis(pyr)))


pyr = im2wv(img,2)
imsave(fn('outputs/prob6a_2.jpg'),clip(vis(pyr)))


pyr = im2wv(img,3)
imsave(fn('outputs/prob6a_3.jpg'),clip(vis(pyr)))

#Inverse transform to reconstruct image
im = clip(wv2im(pyr))
imsave(fn('outputs/prob6b.jpg'),im)

#Zero out some levels and reconstruct
for i in range(len(pyr)-1):

    for j in range(3):
        pyr[i][j][...] = 0.

    im = clip(wv2im(pyr))
    imsave(fn('outputs/prob6b_%d.jpg' % i),im)
