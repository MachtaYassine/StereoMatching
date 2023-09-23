
import numpy as np


class StereoCost:
    
    def pixelwise_AD(intensity_left, intensity_right):
        # Compute the absolute difference between intensities
        abs_diff = np.abs(intensity_left - intensity_right)

        return abs_diff
    
    def pixelwise_SD(intensity_left, intensity_right):
        # Compute the squered difference between intensities
        suqared_diff = (intensity_left - intensity_right)**2

        return suqared_diff
    
    def pixelwise_TAD(intensity_left, intensity_right,threshold):
        # Compute the Truncated absolute difference between intensities
        return np.min(np.abs(intensity_left - intensity_right),threshold)
    
    def NCC(intensites_left, intensities_right):
        # Compute the Truncated absolute difference between intensities
        mean_left = np.mean(intensites_left)
        mean_right = np.mean(intensities_right)
        std_left = np.std(intensites_left)
        std_right = np.std(intensities_right)
        NCC = np.sum((intensites_left - mean_left)*(intensities_right - mean_right))/(std_left*std_right)
        return NCC