import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


left_data_path = "data/images_left"  
right_data_path = "data/images_right"


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
        
# Testing the DataLoader class    
if __name__ == "__main__":
    image_name = "Road"  # Image name
    
    data_loader = DataLoader(image_name)
    left_image, right_image = data_loader.load_images()
    print(type(left_image))
    print(left_image.shape)
    
    if left_image is not None and right_image is not None:
        concatenated_image=np.hstack((left_image,right_image))
        # Create a Matplotlib figure and axis
        fig, ax = plt.subplots()
        # Display the image on the axis
        ax.imshow(concatenated_image, cmap='gray')
        # Turn off axis labels and ticks
        ax.axis('off')
        plt.savefig(os.path.join('utilities/Dataloading testing',f'{image_name}.png'), bbox_inches='tight', pad_inches=0, dpi=300)
        
        pass
    else:
        print("Data loading failed.")        
    