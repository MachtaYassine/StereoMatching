import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import abc
from numba import jit


@jit(nopython=True, parallel=True, cache=True)
def fixed_window_5(cost_function, left_image, right_image, max_disparity,window_size=5):
    height, width = left_image.shape
    disparity_space_image = np.zeros((height, width, max_disparity))
    for y in range(window_size, height - window_size):
        #print progression of the scanlines
        print(int(y/height*100), '%')
        for x in range(window_size, width - window_size):
            for d in range(max_disparity):
                disparity_space_image[y, x, d] = cost_function(left_image, right_image, x, y, d, window_size)
    return disparity_space_image

def fixed_window_11(cost_function, left_image, right_image, max_disparity,window_size=11):
    height, width = left_image.shape
    disparity_space_image = np.zeros((height, width, max_disparity))
    for y in range(window_size, height - window_size):
        #print progression of the scanlines
        print(int(y/height*100), '%')
        for x in range(window_size, width - window_size):
            for d in range(max_disparity):
                disparity_space_image[y, x, d] = cost_function(left_image, right_image, x, y, d, window_size)
    return disparity_space_image


    
    

    

    


