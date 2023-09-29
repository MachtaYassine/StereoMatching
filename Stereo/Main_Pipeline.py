import numpy as np
import cv2
from utilities.load_data import *
from Stereo.Aggregation.aggregation import *
from Stereo.Costs.costs import *
from Stereo.Disparity_computing.disparty_computing import *
from Stereo.Main_Pipeline_jit import *
from numba import jit


def log_array(name,array):
    with open(name+'.txt', "w") as file:
    # Iterate through the elements of the array and write them to the file
        for i,element in enumerate(array):
            if len(array.shape)==3:
                for j,element2 in enumerate(element):
                    file.write(f'all disparity costs for coordinate {i,j} : {element2}' + "\n")
            else:
                file.write(f'values in this row {i} : {element}' + "\n") 


class StereoPipeline:
    '''
    Quick attempt at a modulable class for the stereo pipeline
    '''
    def __init__(self,image_name,
                 cost_function, 
                 aggregation_function,
                 disparity_computation,
                 disparity_refinement,
                 max_disparity=16,
                 occlusion=90,
                 dataloader=DataLoader,
                 datavisualizer=DataVisualizer,):
        

        self.image_name = image_name
        self.dataloader=dataloader
        self.cost_function = cost_function
        self.aggregation_function = aggregation_function
        self.disparity_computation = disparity_computation
        self.disparity_refinment = disparity_refinement
        self.datavisualizer=datavisualizer
        self.max_disparity=max_disparity
        self.occ=occlusion
    def compute_disparity_map(self):
        left_image,right_image=self.dataloader(self.image_name).load_images()
        
        
        disparity_space_image=self.aggregation_function.compute(self.cost_function,left_image, right_image, self.max_disparity)
        log_array(self.image_name+'_disparity_space_image',disparity_space_image)
        disparity_map=self.disparity_computation.compute_disparity_map(disparity_space_image)
        log_array(self.image_name+'_disparity_map',disparity_map)
        if self.disparity_refinment:
            disparity_map=self.disparity_refinment(disparity_map)
            
        method_info=self.cost_function.__name__+'_'+self.aggregation_function.name+'_'+self.disparity_computation.strategy+'_'+str(self.max_disparity)
        DataVisualizer(self.image_name+'_'+method_info+"_disparity_map",disparity_map,None).save_array_as_image(side_to_side=False,plot_image=False)
        return disparity_map
    
    def compute_disparity_with_dp(self):
        
        left_image,right_image=self.dataloader(self.image_name).load_images()
        disparity_space_image=self.aggregation_function.compute(self.cost_function,left_image, right_image, self.max_disparity)
        log_array(self.image_name+'_disparity_space_image_DP',disparity_space_image)
        disparity_left , disparity_right=self.disparity_computation.compute_disparity_map(disparity_space_image,self.occ,self.max_disparity)
        log_array(self.image_name+'_disparity_left_DP',disparity_left)
        log_array(self.image_name+'_disparity_right_DP',disparity_right)
        print(disparity_left.shape)
        method_info=self.cost_function.__name__+'_'+self.aggregation_function.name+'_'+self.disparity_computation.strategy+'_'+str(self.max_disparity)+'_'+ str(self.occ)
        DataVisualizer(self.image_name+'_'+method_info+"_disparity_map",disparity_left,disparity_right).save_array_as_image(side_to_side=False,plot_image=False)
        return disparity_left
    
if __name__=='__main__':
    image_name='chess'
    cost_function=StereoCost.SAD
    aggregation_function=fixed_window(5)
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