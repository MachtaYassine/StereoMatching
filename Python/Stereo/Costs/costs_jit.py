
import numpy as np
from numba import jit


@jit(nopython=True, parallel=True, cache=True)
def SAD(left_intensities, right_intensities,single_pixel=False):
    # we need to map Absolute differences on the to compute the AD element wise between these two arrays
    if single_pixel:
        return np.abs(left_intensities - right_intensities)
    sum=0
    height,width=left_intensities.shape
    for v in range(height):
        for u in range(width):
            sum+=np.abs(left_intensities[v,u] - right_intensities[v,u])
    return sum

@jit(nopython=True, parallel=True, cache=True)
def SSD(left_intensities, right_intensities,single_pixel=False):
    if single_pixel:
        return (left_intensities - right_intensities)**2
    sum=0
    height,width=left_intensities.shape
    for v in range(height):
        for u in range(width):
            sum+=(left_intensities[v,u] - right_intensities[v,u])**2
    return sum
@jit(nopython=True, parallel=True, cache=True)
def STAD(left_intensities, right_intensities,threshold=10):
    # Compute the Truncated absolute difference between intensities
    sum=0
    height,width=left_intensities.shape
    for v in range(height):
        for u in range(width):
            abs_diff=np.abs(left_intensities[v,u] - right_intensities[v,u])
            if abs_diff>threshold:
                abs_diff=threshold
            sum+=abs_diff
    return sum
    
@jit(nopython=True, parallel=True, cache=True)
def NCC(left_intensities, right_intensities):
    # Compute the normaliserd cross correlation difference between intensities
    mean_left=np.mean(left_intensities)
    mean_right=np.mean(right_intensities)
    std_left=np.std(left_intensities)
    std_right=np.std(right_intensities)
    sum=0
    height,width=left_intensities.shape
    for v in range(height):
        for u in range(width):
            sum+=(left_intensities[v, u]-mean_left)*(right_intensities[v, u]-mean_right)
    return sum/(std_left*std_right)
        