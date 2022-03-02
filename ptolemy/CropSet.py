import numpy as np
import pandas as pd
import ptolemy.geometry as geom
from ptolemy.PointSet import PointSet2D
import matplotlib.pyplot as plt

class CropSet:
    def __init__(self, crops, boxes, rotated_boxes):
        self.crops = crops
        self.boxes = boxes
        self.rotated_boxes = rotated_boxes
        # self.crop_ind = np.array(range(len(crops)))
        self.center_coords = PointSet2D.concatenate([PointSet2D([int(np.mean(box.y))], [int(np.mean(box.x))]) for box in boxes])
        center_coords_df = [PointSet2D([int(np.mean(box.y))], [int(np.mean(box.x))]) for box in boxes]
        self.df = pd.DataFrame({'boxes': self.boxes, 'centers': center_coords_df, 'visited': 0})

    def pad(self, width):
        new_crops = []
        for box in self.crops:
            if box.shape[0] > width:
                xcenter = box.shape[0] // 2
                xmin, xmax = xcenter - width//2, xcenter + width//2
                box = box[xmin:xmax, :]
            else:
                left_right_concat = np.zeros(( (width-box.shape[0])//2, box.shape[1]))
                box = np.concatenate((left_right_concat, box, left_right_concat), axis=0)

            if box.shape[1] > width:
                ycenter = box.shape[1] // 2
                ymin, ymax = ycenter - width//2, ycenter + width//2
                box = box[:, ymin:ymax]
            else:
                top_bottom_concat = np.zeros((box.shape[0], (width - box.shape[1])//2))
                box = np.concatenate((top_bottom_concat, box, top_bottom_concat), axis=1)
            
            if box.shape[0] != width:
                box = np.concatenate((np.zeros((1, box.shape[1])), box), axis=0)
            if box.shape[1] != width:
                box = np.concatenate((np.zeros((box.shape[0], 1)), box), axis=1)
            new_crops.append(box)

        self.crops = new_crops

    def normalize_constant(self, mean, std):
        normalized_crops = []
        for crop in self.crops:
            normalized_crops.append((crop - mean) / std)
        self.crops = normalized_crops

    def normalize(self):
        normalized_crops = []
        for crop in self.crops:
            normalized_crops.append((crop - crop.mean()) / crop.std())
        self.crops = normalized_crops
    
    def update_scores(self, scores):
        self.df['prior_scores'] = scores
        self.scores = scores

    def update_finals(self, finals):
        if type(finals) is np.ndarray:
            finals = [arr for arr in finals]
        self.df['final_activations'] = finals
        
    def viz_crops(self):
        for crop in self.crops:
            plt.imshow(crop, cmap='Greys_r')
            plt.axis('off')
            plt.show()
        
    
    

    

        

