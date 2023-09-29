import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import abc
from numba import jit


@jit(nopython=True, parallel=True, cache=True)
def fixed_window(cost_function, left_image, right_image, max_disparity,window_size=5):
    height, width = left_image.shape
    disparity_space_image = np.zeros((height, width, max_disparity))
    for y in range(window_size, height - window_size):
        #print progression of the scanlines
        print(int(y/height*100), '% \r', end='',flush=True)
        for x in range(window_size, width - window_size):
            for d in range(max_disparity):
                left_intensities= left_image[y - window_size:y + window_size + 1, x - window_size:x + window_size + 1]
                right_intensities= right_image[y - window_size:y + window_size + 1, x - window_size - d:x + window_size + 1 - d]
                disparity_space_image[y, x, d] = cost_function(left_intensities,right_intensities)
    return disparity_space_image

@jit(nopython=True, parallel=True, cache=True)
def fixed_window_with_loop(cost_function, left_image, right_image, max_disparity,window_size=5):
    height, width = left_image.shape
    disparity_space_image = np.zeros((height, width, max_disparity))
    for y in range(window_size, height - window_size):
        #print progression of the scanlines
        print(int(y/height*100), '% \r', end='',flush=True)
        for x in range(window_size, width - window_size):
            for d in range(max_disparity):
                for u in range(-window_size,window_size+1):
                    for v in range(-window_size,window_size+1):
                        left_intensity= left_image[y + v, x + u]
                        right_intensity= right_image[y + v, x + u - d]
                        disparity_space_image[y, x, d] += cost_function(left_intensity,right_intensity,single_pixel=True)
    return disparity_space_image


    
    

    

    


