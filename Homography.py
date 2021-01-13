## Default modules imported. Import more if you need to.

import numpy as np


## Fill out these functions yourself

# Fits a homography between pairs of pts
#   pts: Nx4 array of (x,y,x',y') pairs of N >= 4 points
# Return homography that maps from (x,y) to (x',y')
#
# Can use np.linalg.svd
def getH(pts):
    n = pts.shape[0]
    p = pts[:,0:2]
    p = np.concatenate((p, np.ones((n,1))), axis=1)
    p_dash = pts[:,2:4]
    p_dash = np.concatenate((p_dash, np.ones((n,1))), axis=1)
    a_final = np.zeros((3,9))
    for i in range(n):
        hp_i = np.array([[p[i,0],p[i,1],p[i,2],0,0,0,0,0,0],[0,0,0,p[i,0],p[i,1],p[i,2],0,0,0],[0,0,0,0,0,0,p[i,0],p[i,1],p[i,2]]])
        p_i_dash = np.array([[0,-p_dash[i,2],p_dash[i,1]], [p_dash[i,2],0,-p_dash[i,0]], [-p_dash[i,1],p_dash[i,0],0]])
        a = np.matmul(p_i_dash,hp_i)
        if i==0:
            a_final = a
        else:
            a_final = np.concatenate((a_final,a),axis=0)
    
    u,s,v = np.linalg.svd(a_final,full_matrices=True)
    h = np.transpose(v)[:,-1]
    h = np.reshape(h,(3,3))    
    
    return h


# Splices the source image into a quadrilateral in the dest image,
# where dpts in a 4x2 image with each row giving the [x,y] co-ordinates
# of the corner points of the quadrilater (in order, top left, top right,
# bottom left, and bottom right).
#
# Note that both src and dest are color images.
#
# Return a spliced color image.
def splice(src,dest,dpts):
    new_dest = np.copy(dest)
    r_src = src.shape[0]
    c_src = src.shape[1]
    spts = np.array([[0,0],[0,c_src-1], [r_src-1,0], [r_src-1, c_src-1]])
    points = np.append(dpts,spts,axis = 1)
    H = getH(points)
    range_of_points_y =  int((np.max(dpts[:,1])+1) - np.min(dpts[:,1]))
    range_of_points_x = int((np.max(dpts[:,0])+1) - np.min(dpts[:,0]))
    test_array = np.zeros(((range_of_points_x*range_of_points_y),2))

    count = 0
    for i in range(int(np.min(dpts[:,1])),int(np.max(dpts[:,1]))+1):
        for j in range(int(np.min(dpts[:,0])),int(np.max(dpts[:,0]))+1):
            test_array[count,0] = j
            test_array[count,1] = i
            count = count + 1

    range_of_points = np.append(test_array,np.ones((range_of_points_y*range_of_points_x,1)), axis = 1)        
    p_dash = np.matmul(H, np.transpose(range_of_points))
    p_dash[0,:] = p_dash[0,:]/p_dash[2,:]
    p_dash[1,:]  = p_dash[1,:]/p_dash[2,:]
    p_dash = np.delete(p_dash,(2), axis=0)

    pos1 = ((p_dash[0,:] >=0)  *  (p_dash[0,:]<r_src-1))
    pos2 = ((p_dash[1,:] >=0)  *  (p_dash[1,:]<c_src-1))
    pos = (pos1*pos2)
    x_src = p_dash[0,np.where(pos==True)]
    y_src = p_dash[1,np.where(pos==True)]
    x_dest = range_of_points[np.where(pos==True),0]
    y_dest = range_of_points[np.where(pos==True),1]

    src_x_floor, src_y_floor = np.int32(np.floor(x_src)), np.int32(np.floor(y_src))
    src_xc, src_yc = src_x_floor+1, src_y_floor+1
    diffx_f =  (x_src - src_x_floor).T
    diffy_f =  (y_src - src_y_floor).T

    for i in range(x_src.shape[1]):
        test = (diffy_f[i,0]*(diffx_f[i,0]*src[src_x_floor[0,i],src_y_floor[0,i],:] + diffy_f[i,0]*src[src_x_floor[0,i],src_yc[0,i],:]) + diffy_f[i,0]*(diffx_f[i,0]*src[src_xc[0,i],src_y_floor[0,i],:] + diffy_f[i,0]*src[src_xc[0,i],src_yc[0,i],:]))
        new_dest[int(y_dest[0,i]),int(x_dest[0,i]),:] = test
    return new_dest


    
########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


simg = np.float32(imread(fn('inputs/p4src.png')))/255.
dimg = np.float32(imread(fn('inputs/p4dest.png')))/255.
dpts = np.float32([ [276,54],[406,79],[280,182],[408,196]]) # Hard coded
    
comb = splice(simg,dimg,dpts)
#comb = np.maximum(0, comb)
imsave(fn('outputs/prob4.png'),comb)
