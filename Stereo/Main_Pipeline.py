import numpy as np
import cv2
from utilities.load_data import *
from Stereo.Aggregation.aggregation import *
from Stereo.Costs.costs import *
from Stereo.Disparity_computing.disparty_computing import *
from Stereo.Main_Pipeline import *


class StereoPipeline:
    def __init__(self,image_name,
                 cost_function, 
                 aggregation_function,
                 disparity_computation,
                 disparity_refinement,
                 dataloader=DataLoader,
                 datavisualizer=DataVisualizer):
        
        
        self.image_name = image_name
        self.dataloader=dataloader
        self.cost_function = cost_function
        self.aggregation_function = aggregation_function
        self.disparity_computation = disparity_computation
        self.disparity_refinment = disparity_refinement
        self.datavisualizer=datavisualizer
        
        
    def compute_disparity_map(self,max_disparity=16):
        left_image,right_image=self.dataloader(self.image_name).load_images()
        
        
        disparity_space_image=self.aggregation_function.compute(self.cost_function,left_image, right_image, max_disparity)
        
        disparity_map=self.disparity_computation(disparity_space_image)
        if self.disparity_refinment:
            disparity_map=self.disparity_refinment(disparity_map)
        DataVisualizer(self.image_name+"_disparity_map",left_image,None).save_array_as_image(side_to_side=False,plot_image=True)
        return disparity_map
    
    
    
    
if __name__=='__main__':
    image_name='chess'
    cost_function=StereoCost.SAD
    aggregation_function=fixed_window(5)
    disparity_computation=DisparityComputation()
    disparity_refinement=None
    StereoPipeline(image_name,
                 cost_function, 
                 aggregation_function,
                 disparity_computation,
                 disparity_refinement,
                 dataloader=DataLoader,
                 datavisualizer=DataVisualizer).compute_disparity_map()