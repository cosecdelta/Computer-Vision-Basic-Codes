## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave
from matplotlib import pyplot as plt
# Fill this out
# X is input 8-bit grayscale image
# Return equalized image with intensities from 0-255
def histeq(X):
    values, counts = np.unique(X, return_counts=True)
    cumulative_sum = np.cumsum(counts)
    max_freq = cumulative_sum[len(values)-1]
    cdf = (cumulative_sum/max_freq)*255.
    round_cdf = np.around(cdf)
    new_image = np.zeros(img.size)
    value_finder = dict(zip(values,round_cdf))
    for i,j in enumerate(X.flat):
        new_image[i] = value_finder[j]
    new_image = np.reshape(new_image, X.shape)
    return new_image
    

########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img = imread(fn('inputs/p2_inp.jpg'))
out = histeq(img)
out = np.maximum(0,np.minimum(255,out))
out = np.uint8(out)
imsave(fn('outputs/prob2.jpg'),out)

#### For Plotting the Intensity Histograms #################

# plt.hist(img.flatten(),256,[0,256])
# plt.xlim([0,256])
# plt.title('Intensity Histogram before Histogram Equalization')
# plt.show()

# plt.hist(out.flatten(),256,[0,256])
# plt.xlim([0,256])
# plt.title('Intensity Histogram after Histogram Equalization')
# plt.show()
