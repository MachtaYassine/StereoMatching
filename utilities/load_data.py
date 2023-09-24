import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


left_data_path = "data/images_left"  
right_data_path = "data/images_right"
results_path = "data/results"
testing_path='utilities/Dataloading_testing'


class DataLoader:
    def __init__(self, image_name):
        self.left_image_path = os.path.join(left_data_path,image_name+"_left.png") 
        self.right_image_path = os.path.join(right_data_path,image_name+"_right.png")

    def load_images(self):
        try:
            # Load left and right images
            left_image = cv2.imread(self.left_image_path, cv2.IMREAD_GRAYSCALE)
            right_image = cv2.imread(self.right_image_path, cv2.IMREAD_GRAYSCALE)

            if left_image is None or right_image is None:
                raise FileNotFoundError("Could not load one or both of the images.")

            return left_image, right_image
        except Exception as e:
            print("Error loading images:", str(e))
            return None, None



def save_fig(image,name,save_path,plot_image):
    # Create a figure and axis for the left image
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.axis('off')

    # Save the left image
    plt.savefig(os.path.join(save_path,name+'.png'), bbox_inches='tight', pad_inches=0, dpi=300)
    if plot_image:
        plt.show()
    plt.close(fig)  # Close the figure to release resources
class DataVisualizer:
    def __init__(self,name,array_left,array_right) :
        self.array_left = array_left
        self.array_right = array_right
        self.name = name
    def save_array_as_image(self,side_to_side=True,testing=False,plot_image=False):
        if testing:
            save_path=testing_path
        else:
            save_path=results_path
        if left_image is not None and right_image is not None:
            if side_to_side:
                #Create a Matplotlib figure and axis
                fig, ax = plt.subplots(1,2)
                # Display the image on the axis
                ax[0].imshow(left_image, cmap='gray')
                ax[1].imshow(right_image, cmap='gray')
                # Turn off axis labels and ticks
                ax[0].axis('off')
                ax[1].axis('off')
                plt.savefig(os.path.join(save_path,f'{self.name}.png'), bbox_inches='tight', pad_inches=0, dpi=300)
            else:
                save_fig(self.array_left,self.name+'_left',save_path,plot_image)
                save_fig(self.array_right,self.name+'_right',save_path,plot_image)
                
        elif left_image is not None:
            print("Single image mode")
            save_fig(self.array_left,self.name,save_path,plot_image)
            
    
    
    
    
    
    
# Testing the DataLoader class    
if __name__ == "__main__":
    image_name = "Road"  # Image name
    
    data_loader = DataLoader(image_name)
    left_image, right_image = data_loader.load_images()
    DataVisualizer(image_name,left_image,right_image).save_array_as_image(side_to_side=False)
    
            
    