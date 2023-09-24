import numpy as np
from tqdm import tqdm

class fixed_window:
    def __init__(self,window_size):
        self.window_size=window_size
    
    
    def compute(self,cost_fuction,left_image,right_image,max_disparity):
        '''
        computes the aggregated costs of fixed window centered at x,y in the left image and x-d,y in the right image

        '''
        # initialize the disparity space image
        width,height=left_image.shape
        print(width,height)
        disparity_space_image=np.zeros((width,height,max_disparity))
        #define border exculsuion to avoid the window to go out of the image
        for y in tqdm(range(self.window_size//2,height-self.window_size//2)):
            for x in tqdm(range(self.window_size//2,width-self.window_size//2)):
                #compute the cost for each disparity level
                for d in range(0,max_disparity):
                    #compute the cost for each disparity level
                    window_left=left_image[y-self.window_size//2:y+self.window_size//2+1,x-self.window_size//2:x+self.window_size//2+1]
                    window_right=right_image[y-self.window_size//2:y+self.window_size//2+1,x-self.window_size//2-d:x+self.window_size//2+1-d]
                    if window_right.shape==window_left.shape:
                        disparity_space_image[y,x,d]=cost_fuction(window_left,window_right)
        
    
        return disparity_space_image
    


