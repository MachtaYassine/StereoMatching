
import numpy as np
from numba import jit


@jit(nopython=True, parallel=True, cache=True)
def SAD(left_image, right_image, x, y, d, window_size):
    # we need to map Absolute differences on the to compute the AD element wise between these two arrays
    sum=0
    for v in range(-window_size, window_size + 1):
        for u in range(-window_size, window_size + 1):
            sum+=np.abs(left_image[y+v, x+u] - right_image[y+v, x+u-d])
    return sum

@jit(nopython=True, parallel=True, cache=True)
def SSD(left_image, right_image, x, y, d, window_size):
    sum=0
    for v in range(-window_size, window_size + 1):
        for u in range(-window_size, window_size + 1):
            sum+=(left_image[y+v, x+u] - right_image[y+v, x+u-d])**2
            
    return sum
@jit(nopython=True, parallel=True, cache=True)
def STAD(left_image, right_image, x, y, d, window_size,threshold=10):
    # Compute the Truncated absolute difference between intensities
    sum=0
    for v in range(-window_size, window_size + 1):
        for u in range(-window_size, window_size + 1):
            abs_diff=np.abs(left_image[y+v, x+u] - right_image[y+v, x+u-d])
            if abs_diff<threshold:
                sum+=abs_diff
    return sum
    
@jit(nopython=True, parallel=True, cache=True)
def NCC(left_image, right_image, x, y, d, window_size):
    # Compute the normaliserd cross correlation difference between intensities
    mean_left=np.mean(left_image[y-window_size:y+window_size,x-window_size:x+window_size])
    mean_right=np.mean(right_image[y-window_size:y+window_size,x-d-window_size:x-d+window_size])
    std_left=np.std(left_image[y-window_size:y+window_size,x-window_size:x+window_size])
    std_right=np.std(right_image[y-window_size:y+window_size,x-d-window_size:x-d+window_size])
    sum=0
    for v in range(-window_size, window_size + 1):
        for u in range(-window_size, window_size + 1):
            sum+=(left_image[y+v, x+u]-mean_left)*(right_image[y+v, x+u-d]-mean_right)
    return sum/(std_left*std_right)
        