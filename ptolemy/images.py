# Combine image w/ MRC
from __future__ import absolute_import

import numpy as np
from PIL import Image
import ptolemy.mrc as mrc
from ptolemy.algorithms import flood_segments
import copy
from ptolemy.CropSet import CropSet
import matplotlib
import matplotlib.pyplot as plt


def load_mrc(path):
    with open(path, 'rb') as f:
        content = f.read()
    im,_,_ = mrc.parse(content)
    im = np.copy(im)
    return im


class Exposure:
    def __init__(self, image, scale=1, operator_selections=None):
        # image is assumed to be unrotated, boxes also unrotated

        if np.sum(image < 0) != 0:
            image[image < 0] = 0

        self.image = image
        self.scale = scale

        self.size = image.shape
        # self.image_scaled = image
        # if scale > 1:
        #     n = int(np.round(image.shape[0]/scale))
        #     m = int(np.round(image.shape[1]/scale))
        #     self.size = (n,m)
        #     self.image_scaled = downsample(image, self.size)
        if operator_selections:
            self.operator_selections = operator_selections

    def make_mask(self, mask_alg):
        self.mask = mask_alg.forward(self.image)

        # self.boxes, self.rotated_boxes, self.rotated_image, self.rot_ang_deg, self.mask = segment_alg.forward(self.image)
        # self.box_areas = [box.area() for box in self.boxes]
        # self.segment_alg = segment_alg
        # return copy.deepcopy(self.boxes), copy.deepcopy(self.rotated_boxes), copy.deepcopy(self.rotated_image), copy.deepcopy(self.rot_ang_deg)

    def process_mask(self, process_alg):
        self.boxes, self.rotated_boxes, self.rotated_image, self.rot_ang_deg, self.mean_intensities = process_alg.forward(self.mask, self.image)

    def get_crops(self, crop_alg=None, min_width=None):
        if not hasattr(self, 'rotated_boxes'):
            raise ValueError
            
        crops = []
        boxes = []
        rotated_boxes = []
        for box, rotated_box in zip(self.boxes, self.rotated_boxes):
            segmented_box = self.rotated_image[int(max(rotated_box.xmin(), 0)): int(min(rotated_box.xmax(), self.rotated_image.shape[0])) ,
                                               int(max(rotated_box.ymin(), 0)): int(min(rotated_box.ymax(), self.rotated_image.shape[1])) ]
            if min_width and (segmented_box.shape[0] < min_width or segmented_box.shape[1] < min_width):
                continue
            crops.append(segmented_box)
            boxes.append(box)
            rotated_boxes.append(rotated_box)
            
        self.boxes = boxes
        self.rotated_boxes = rotated_boxes

        crops = CropSet(crops, self.boxes, self.rotated_boxes)

        if crop_alg is not None:
            crops = crop_alg.forward(self, crops)
        
        self.crops = crops
        return crops

    def viz_boxes(self, rotated=False, selections=False, given=False):
        if selections and not hasattr(self, 'operator_selections'):
            raise ValueError
        
        if rotated:
            image_to_show = self.rotated_image
            boxes_to_show = self.rotated_boxes
        else:
            image_to_show = self.image
            boxes_to_show = self.boxes
            
        _, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(image_to_show, cmap='Greys_r')
        patches = []
        
        if given:
            boxes_to_show = given
        for box in boxes_to_show:
            patches.append(matplotlib.patches.Polygon(box.as_matrix_y(), facecolor='None'))
        collection = matplotlib.collections.PatchCollection(patches)
        ax.add_collection(collection)
        collection.set_color('r')
        collection.set_facecolor('none')
        collection.set_linewidth(2)
        plt.axis('off')
        
        if selections:
            plt.scatter(self.operator_selections.x, self.operator_selections.y)
        
        # plt.show()
        # plt.plot image and boxes 
        

    def viz_boxes_and_scores(self, rotated=False, selections=False, numeric_scores=False):
        # plt.plot image and boxes with nice viz
        if not hasattr(self, 'boxes'):
            raise ValueError # say you haven't gotten the boxes yet
        if 'prior_scores' not in self.crops.df.columns:
            raise ValueError # say you haven't scored the crops yet
        if selections and not hasattr(self, 'operator_selections'):
            raise ValueError
        
        if rotated:
            image_to_show = self.rotated_image
            boxes_to_show = self.crops.rotated_boxes
        else:
            image_to_show = self.image
            boxes_to_show = self.crops.boxes
            
        cmap = plt.get_cmap('RdYlBu')
        _, ax = plt.subplots(figsize=(12,12))
        ax.imshow(image_to_show, cmap='Greys_r')
        
        patches = []
        colors = []
        
        for box, score in zip(boxes_to_show, self.crops.df.prior_scores):
            patches.append(matplotlib.patches.Polygon(box.as_matrix_y(), facecolor='none'))
            colors.append(score)
            
        collection = matplotlib.collections.PatchCollection(patches)
        ax.add_collection(collection)
        colors = np.array(colors)
        collection.set_color(cmap(colors / np.max(colors)))
        collection.set_facecolor('none')
        collection.set_linewidth(2)
        plt.axis('off')
        if selections:
            plt.scatter(self.operator_selections.x, self.operator_selections.y)
            
        if numeric_scores:
            for center, text in zip(self.crops.df.centers, self.crops.df.prior_scores):
                plt.text(center.y[0]+25, center.x[0]-25, str(round(text, 2)), color='red', bbox=dict(facecolor='white'))
        
        # plt.show()

        
    def viz_mask(self, selections=False, imsize=(8, 8)):
        plt.figure(figsize=imsize)
        plt.imshow(self.mask, cmap='Greys_r')
        plt.axis('off')
        if selections:
            plt.scatter(self.operator_selections.x, self.operator_selections.y)
        # plt.show()
        
    def viz_image(self, selections=False, imsize=(8, 8)):
        plt.figure(figsize=imsize)
        plt.imshow(self.image, cmap='Greys_r')
        plt.axis('off')
        if selections:
            plt.scatter(self.operator_selections.x, self.operator_selections.y)
        # plt.show()
        
        
    def viz_image_centers(self, imsize=(8, 8)):
        plt.figure(figsize=imsize)
        plt.imshow(self.image, cmap='Greys_r')
        plt.scatter(self.crops.center_coords.y, self.crops.center_coords.x)
        plt.axis('off')

    def score_crops(self, classifier, final):
        if not final:
            scores = classifier.forward_cropset(self.crops)
            self.crops.update_scores(scores)
        else:
            scores, finals = classifier.forward_cropset(self.crops)
            self.crops.update_scores(scores)
            self.crops.update_finals(finals)


    # def get_crops(self, postprocess_alg=None):
    #     if not hasattr(self, 'rotated_boxes'):
    #         raise ValueError
    #     crops = []
    #     for box in self.rotated_boxes:
    #         segmented_box = self.rotated_image[max(box.ymin(), 0): min(box.ymax(), self.rotated_image.shape[0]) ,
    #                                            max(box.xmin(), 0): min(box.xmax(), self.rotated_image.shape[1]) ]
    #         crops.append(segmented_box)

    #     crops = CropSet2D(crops, self.boxes, self.rotated_boxes)

    #     if postprocess_alg is not None:
    #         crops = postprocess_alg.forward(self, crops)
        
    #     self.crops = crops
    #     return crops


# class Medium_Mag_Exposure:
#     def __init__(self, image, operator_selections=None):
#         self.image = image
#         self.size = image.shape
#         self.operator_selections = operator_selections

#     def segment(self, segment_alg):
#         self.mask = segment_alg.forward(self.image)

#     def process_mask(self, )
    
#     def get_crops(self, postprocess_alg):
#         self.boxes, self.rotated_boxes, self.rotated_image, self.rot_ang_deg = postprocess_alg.forward(self.mask)


#     def viz_boxes(self, rotated=False, selections=False):
#         # plt.plot image and boxes 
#         raise NotImplementedError

#     def viz_boxes_and_scores(self, rotated=False, selections=False):
#         # plt.plot image and boxes with nice viz
#         raise NotImplementedError

    
