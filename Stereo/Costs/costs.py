
import numpy as np


class StereoCost:
    
    def SAD(intensites_left, intensities_right):
        # we need to map Absolute differences on the to compute the AD element wise between these two arrays
        return np.sum(np.abs(intensites_left-intensities_right))

    
    def SSD(intensites_left, intensities_right):
        # Compute the squered difference between intensities

        return np/sum((intensites_left - intensities_right)**2)

    
    def STAD(intensites_left, intensities_right,threshold):
        # Compute the Truncated absolute difference between intensities
        AD=np.abs(intensites_left - intensities_right)
        return np.sum(np.where(AD>threshold,threshold,AD))
    
    def NCC(intensites_left, intensities_right):
        # Compute the Truncated absolute difference between intensities
        mean_left = np.mean(intensites_left)
        mean_right = np.mean(intensities_right)
        std_left = np.std(intensites_left)
        std_right = np.std(intensities_right)
        NCC = np.sum((intensites_left - mean_left)*(intensities_right - mean_right))/(std_left*std_right)
        return NCC