import numpy as np
import cv2
from utilities.load_data import *
from Stereo.Aggregation.aggregation_jit import *
from Stereo.Costs.costs import *
from Stereo.Disparity_computing.disparty_computing import *
from Stereo.Main_Pipeline_jit import *
from numba import jit


def log_array(name,array):
    with open(name+'.txt', "w") as file:
    # Iterate through the elements of the array and write them to the file
        for element in array:
            file.write(f'values in this row {np.unique(element)}' + "\n") 

class StereoPipeline:
    def __init__(self,image_name,
                 cost_function, 
                 aggregation_function,
                 disparity_computation,
                 disparity_refinement,
                 max_disparity=64,
                 dataloader=DataLoader,
                 datavisualizer=DataVisualizer):
        
        
        self.image_name = image_name
        self.dataloader=dataloader
        self.cost_function = cost_function
        self.aggregation_function = aggregation_function
        self.disparity_computation = disparity_computation
        self.disparity_refinment = disparity_refinement
        self.datavisualizer=datavisualizer
        self.max_disparity=max_disparity
        
    
    def compute_disparity_map(self):
        left_image,right_image=self.dataloader(self.image_name).load_images()
        
        
        disparity_space_image=self.aggregation_function(self.cost_function,left_image, right_image, self.max_disparity)
        log_array('DSI',disparity_space_image)
        disparity_map=self.disparity_computation().compute_disparity_map(disparity_space_image)
        log_array('disparity_map',disparity_map)
        if self.disparity_refinment:
            disparity_map=self.disparity_refinment(disparity_map)
        DataVisualizer(self.image_name+'_'+self.cost_function.__name__+'_'+self.aggregation_function.__name__+'_'+self.disparity_computation.__name__+'_'+str(self.max_disparity)+"_disparity_map",disparity_map,None).save_array_as_image(side_to_side=False,plot_image=False)
        return disparity_map
    
    
    
    
if __name__=='__main__':
    image_name='chess'
    cost_function=StereoCost.SAD
    aggregation_function=fixed_window_5
    disparity_computation=DisparityComputation()
    disparity_refinement=None
    StereoPipeline(image_name,
                   'fixed_window_5_SAD_winner_takes_all',
                 cost_function, 
                 aggregation_function,
                 disparity_computation,
                 disparity_refinement,
                 dataloader=DataLoader,
                 datavisualizer=DataVisualizer).compute_disparity_map()