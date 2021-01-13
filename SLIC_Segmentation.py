## Default modules imported. Import more if you need to.
### Problem designed by Abby Stylianou

import numpy as np
from scipy.signal import convolve2d as conv2

def get_cluster_centers(im,num_clusters):
    # Implement a method that returns an initial grid of cluster centers. You should first
    # create a grid of evenly spaced centers (hint: np.meshgrid), and then use the method
    # discussed in class to make sure no centers are initialized on a sharp boundary.
    # You can use the get_gradients method from the support code below.
    
    
    """ YOUR CODE GOES HERE """
    cluster_centers = np.zeros((num_clusters,2),dtype='int') 
    h = im.shape[0]
    w = im.shape[1]
    s = np.sqrt((h*w)/num_clusters)
    row_s = np.around(h/s)
    col_s = np.around(w/s)
    rows = np.linspace(1,h,int(row_s),endpoint=False)
    cols = np.linspace(1,w,int(col_s),endpoint=False)
    cluster = np.meshgrid(cols,rows,indexing='ij')
    cluster_centers[:,0] = cluster[0].flatten()
    cluster_centers[:,1] = cluster[1].flatten()
    grad_img = get_gradients(im)
    for i in range(cluster_centers.shape[0]):
        x = cluster_centers[i,0]
        y = cluster_centers[i,1]
        neighbour = grad_img[x-1:x+2,y-1:y+2]
        minimum_el = np.argmin(neighbour)
        x_cor = minimum_el/3
        y_cor = minimum_el % 3
        cluster_centers[i,0] = x + (x_cor-1)
        cluster_centers[i,1] = y + (y_cor-1)
    
    

    return cluster_centers

def slic(im,num_clusters,cluster_centers):
    # Implement the slic function such that all pixels assigned to a label
    # should be close to each other in squared distance of augmented vectors.
    # You can weight the color and spatial components of the augmented vectors
    # differently. To do this, experiment with different values of spatial_weight.
    h = im.shape[0]
    w = im.shape[1]
    s = np.around(np.sqrt((h*w)/num_clusters))
    coordinates = np.meshgrid(np.arange(0,h), np.arange(0,w),indexing='ij')
    alpha = 1.5
    img_aug = np.dstack((im,alpha*coordinates[0], alpha*coordinates[1]))
    mu = img_aug[cluster_centers[:,0], cluster_centers[:,1],:]
    dist_min = np.ones((h,w)) * np.inf
    label = np.ones((h,w))
    label_update = np.copy(label)



    flag=1
    while(flag==1):
        for i in range(num_clusters):
            x_point = cluster_centers[i,0]
            y_point = cluster_centers[i,1]
            x_min = np.maximum(x_point-s,0)
            x_max = np.minimum(x_point+s,w-1)
            y_min = np.maximum(y_point-s,0)
            y_max = np.minimum(y_point+s,h-1)
            neighbor_aug = img_aug[int(x_min):int(x_max), int(y_min):int(y_max),:]
            neighbor_label = label[int(x_min):int(x_max), int(y_min):int(y_max)]
            dist_mat = dist_min[int(x_min):int(x_max), int(y_min):int(y_max)]
            mu_val = mu[i,:]
            mu_i = np.tile(mu_val,(neighbor_aug.shape[0], neighbor_aug.shape[1],1))
            dist_temp = np.sum((neighbor_aug - mu_i)**2,axis=2)
            neighbor_label[dist_temp<dist_mat] = i
            dist_min[int(x_min):int(x_max), int(y_min):int(y_max)] = np.minimum(dist_mat,dist_temp)
            label[int(x_min):int(x_max), int(y_min):int(y_max)] = neighbor_label
            update = np.where(neighbor_label==i)
            update_neighbor = neighbor_aug[update[0], update[1]]
            mean_update = np.mean(update_neighbor,axis = 0)
            mu[i,:] = mean_update
            cluster_centers[i,0] = mean_update[3]
            cluster_centers[i,1] = mean_update[4]
            flag = flag + 1
            if np.equal(label_update.all(), label.all()):
                flag = 0
            else:
                flag = 1
                label_update = np.copy(label)

    """ YOUR CODE GOES HERE """
    return label


########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

# Use get_gradients (code from pset1) to get the gradient of your image when initializing your cluster centers.
def get_gradients(im):
    if len(im.shape) > 2:
        im = np.mean(im,axis=2)
    df = np.float32([[1,0,-1]])
    sf = np.float32([[1,2,1]])
    gx = conv2(im,sf.T,'same','symm')
    gx = conv2(gx,df,'same','symm')
    gy = conv2(im,sf,'same','symm')
    gy = conv2(gy,df.T,'same','symm')
    return np.sqrt(gx*gx+gy*gy)

# normalize_im normalizes our output to be between 0 and 1
def normalize_im(im):
    im += np.abs(np.min(im))
    im /= np.max(im)
    return im

# create an output image of our cluster centers
def create_centers_im(im,centers):
    for center in centers:
        im[center[0]-2:center[0]+2,center[1]-2:center[1]+2] = [255.,0.,255.]
    return im

im = np.float32(imread(fn('inputs/lion.jpg')))


num_clusters = [25,49,64,81,100]
for num_clusters in num_clusters:
    cluster_centers = get_cluster_centers(im,num_clusters)
    imsave(fn('outputs/prob1a_' + str(num_clusters)+'_centers.jpg'),normalize_im(create_centers_im(im.copy(),cluster_centers)))
    out_im = slic(im,num_clusters,cluster_centers)

    Lr = np.random.permutation(num_clusters)
    out_im = Lr[np.int32(out_im)]
    dimg = cm.jet(np.minimum(1,np.float32(out_im.flatten())/float(num_clusters)))[:,0:3]
    dimg = dimg.reshape([out_im.shape[0],out_im.shape[1],3])
    imsave(fn('outputs/prob1b_'+str(num_clusters)+'.jpg'),normalize_im(dimg))
