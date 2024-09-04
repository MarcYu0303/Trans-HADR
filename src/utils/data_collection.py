import os, re
import cv2
import numpy as np

class DataCollection(object):
    def __init__(self, path = '/home/ssr/transparent/data/0718/1'):
        self.rgb_path = os.path.join(path, 'rgb')
        self.depth_path = os.path.join(path, 'depth')
        self.depth_image_path = os.path.join(path, 'depth_image')
        if not os.path.exists(self.rgb_path):
            os.makedirs(self.rgb_path)
        if not os.path.exists(self.depth_path):
            os.makedirs(self.depth_path)
        if not os .path.exists(self.depth_image_path):
            os.makedirs(self.depth_image_path)
            
        self.index = self.find_next_image_index(self.rgb_path)
            
    def find_next_image_index(folder_path, file_extension="png"):
        # List all files in the folder
        files = os.listdir(folder_path)
        
        # Extract the numeric parts of the filenames
        indices = [0]
        for file in files:
            if file.endswith(file_extension):
                match = re.match(r"(\d+)\." + file_extension, file)
                if match:
                    indices.append(int(match.group(1)))
        
        # Find the next index
        if indices:
            next_index = max(indices) + 1
        else:
            next_index = 1
    
        return next_index

    def get_data(self, color_image, depth_image):
        cv2.imwrite(os.path.join(self.rgb_path, f"{self.index}.png"), color_image)
        cv2.imwrite(os.path.join(self.depth_image_path, f"{self.index}.png"), depth_image)
        np.save(os.path.join(self.depth_path, f"{self.index}.npy"), depth_image)
        self.index += 1
        print(f"Saved image {self.index - 1}")