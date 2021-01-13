## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave



## Fill out these functions yourself

## Take color image, and return 'white balanced' color image
## based on gray world, as described in Problem 2(a). For each
## channel, find the average intensity across all pixels.
##
## Now multiply each channel by multipliers that are inversely
## proportional to these averages, but add upto 3.
def balance2a(img):
    img_new = np.copy(img)
    red_channel_mean = np.mean(img[:,:,0])
    green_channel_mean = np.mean(img[:,:,1])
    blue_channel_mean = np.mean(img[:,:,2])
    # print("Red Mean",red_channel_mean)
    # print("Blue Mean",blue_channel_mean)
    # print("Green Mean",green_channel_mean)
    
    
    ## Calculating the scalars as the reciprocals of the mean intensity in each channel
    alpha_red = 1/red_channel_mean
    alpha_green = 1/green_channel_mean
    alpha_blue = 1/blue_channel_mean
    scalar = 3/(alpha_blue + alpha_green + alpha_red)
    alpha_red = alpha_red*scalar
    alpha_green = alpha_green*scalar
    alpha_blue = alpha_blue*scalar
    
    img_new[:,:,0] = np.multiply(img[:,:,0],alpha_red)
    img_new[:,:,1] = np.multiply(img[:,:,1],alpha_green)
    img_new[:,:,2] = np.multiply(img[:,:,2],alpha_blue)
    
    
    ## Calculated the mean of the new transformed image for each channel. In Gray world color constancy the mean is each channel are equal
    
    # mean_red = np.mean(img_new[:,:,0])
    # mean_green = np.mean(img_new[:,:,1])
    # mean_blue = np.mean(img_new[:,:,2])   
    # print("Red Mean",mean_red)        
    # print("Blue Mean",mean_blue)
    # print("Green Mean",mean_green)
    

    return img_new


## Take color image, and return 'white balanced' color image
## based on description in Problem 2(b). In each channel, find
## top 10% of the brightest intensities, take their average.
##
## Now multiply each channel by multipliers that are inversely
## proportional to these averages, but add upto 3.
def balance2b(img):
    img_new = np.copy(img)
    red_channel_sort = np.sort(img[:,:,0].flatten())[::-1]
    green_channel_sort = np.sort(img[:,:,1].flatten())[::-1]
    blue_channel_sort = np.sort(img[:,:,2].flatten())[::-1]
    ten_percent_index = img[:,:,0].size//10
    
    ## Calculating the scalars as the reciprocals of the mean intensity in each channel for top 10% pixels
    red_channel_mean = np.mean(red_channel_sort[0:ten_percent_index])
    green_channel_mean = np.mean(green_channel_sort[0:ten_percent_index])
    blue_channel_mean = np.mean(blue_channel_sort[0:ten_percent_index])
    # print("Red Mean for top 10%",red_channel_mean)
    # print("Blue Mean for top 10%",blue_channel_mean)
    # print("Green Mean for top 10%",green_channel_mean)
    
    alpha_red = 1/red_channel_mean
    alpha_green = 1/green_channel_mean
    alpha_blue = 1/blue_channel_mean
    scalar = 3/(alpha_blue + alpha_green + alpha_red)
    alpha_red = alpha_red*scalar
    alpha_green = alpha_green*scalar
    alpha_blue = alpha_blue*scalar
    
    img_new[:,:,0] = img[:,:,0] * alpha_red
    img_new[:,:,1] = img[:,:,1] * alpha_green
    img_new[:,:,2] = img[:,:,2] * alpha_blue
    
    ## Calculated the mean of the new transformed image for each channel. 
    ## When we perform the brightest pixel assumption color constancy the image highlights the color of the light source while maintaining the color balance
    
    # mean_red = np.mean(img_new[:,:,0])
    # mean_green = np.mean(img_new[:,:,1])
    # mean_blue = np.mean(img_new[:,:,2])
    # print("Red Mean after color constancy",mean_red)
    # print("Blue Mean after color constancy",mean_blue)
    # print("Green Mean after color constancy",mean_green)
    
    return img_new


########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))



############# Main Program
im1 = np.float32(imread(fn('inputs/CC/ex1.jpg')))/255.
im2 = np.float32(imread(fn('inputs/CC/ex2.jpg')))/255.
im3 = np.float32(imread(fn('inputs/CC/ex3.jpg')))/255.

im1a = balance2a(im1)
im2a = balance2a(im2)
im3a = balance2a(im3)

imsave(fn('outputs/prob2a_1.png'),clip(im1a))
imsave(fn('outputs/prob2a_2.png'),clip(im2a))
imsave(fn('outputs/prob2a_3.png'),clip(im3a))

im1b = balance2b(im1)
im2b = balance2b(im2)
im3b = balance2b(im3)

imsave(fn('outputs/prob2b_1.png'),clip(im1b))
imsave(fn('outputs/prob2b_2.png'),clip(im2b))
imsave(fn('outputs/prob2b_3.png'),clip(im3b))
