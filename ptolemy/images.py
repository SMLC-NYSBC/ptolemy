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
        
        self.operator_selections = operator_selections

    def make_mask(self, mask_alg):
        self.mask = mask_alg.forward(self.image)

        # self.boxes, self.rotated_boxes, self.rotated_image, self.rot_ang_deg, self.mask = segment_alg.forward(self.image)
        # self.box_areas = [box.area() for box in self.boxes]
        # self.segment_alg = segment_alg
        # return copy.deepcopy(self.boxes), copy.deepcopy(self.rotated_boxes), copy.deepcopy(self.rotated_image), copy.deepcopy(self.rot_ang_deg)

    def process_mask(self, process_alg):
        self.boxes, self.rotated_boxes, self.rotated_image, self.rot_ang_deg = process_alg.forward(self.mask, self.image)

    def get_crops(self, crop_alg=None):
        if not hasattr(self, 'rotated_boxes'):
            raise ValueError
        crops = []
        for box in self.rotated_boxes:
            segmented_box = self.rotated_image[int(max(box.xmin(), 0)): int(min(box.xmax(), self.rotated_image.shape[0])) ,
                                               int(max(box.ymin(), 0)): int(min(box.ymax(), self.rotated_image.shape[1])) ]
            if segmented_box.size < 100 or segmented_box.max() == segmented_box.min():
                continue
            crops.append(segmented_box)

        crops = CropSet(crops, self.boxes, self.rotated_boxes)

        if crop_alg is not None:
            crops = crop_alg.forward(self, crops)
        
        self.crops = crops
        return crops

    def viz_boxes(self, rotated=False, selections=False):
        if selections:
            raise NotImplementedError
        
        if rotated:
            image_to_show = self.rotated_image
            boxes_to_show = self.rotated_boxes
        else:
            image_to_show = self.image
            boxes_to_show = self.boxes
            
        _, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(image_to_show, cmap='Greys_r')
        patches = []
        for box in boxes_to_show:
            patches.append(matplotlib.patches.Polygon(box.as_matrix_y(), facecolor='None'))
        collection = matplotlib.collections.PatchCollection(patches)
        ax.add_collection(collection)
        collection.set_color('r')
        collection.set_facecolor('none')
        collection.set_linewidth(2)
        plt.axis('off')
        plt.show()
        # plt.plot image and boxes 

    def viz_boxes_and_scores(self, rotated=False, selections=False):
        # plt.plot image and boxes with nice viz
        if not hasattr(self, 'boxes'):
            raise ValueError # say you haven't gotten the boxes yet
        if 'scores' not in self.crops.df.columns:
            raise ValueError # say you haven't scored the crops yet
        if selections:
            raise NotImplementedError
        
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
        
        for box, score in zip(boxes_to_show, self.crops.df.scores):
            patches.append(matplotlib.patches.Polygon(box.as_matrix_y(), facecolor='none'))
            colors.append(score)
            
        collection = matplotlib.collections.PatchCollection(patches)
        ax.add_collection(collection)
        colors = np.array(colors)
        collection.set_color(cmap(colors / np.max(colors)))
        collection.set_facecolor('none')
        collection.set_linewidth(2)
        plt.axis('off')
        plt.show()

        
    def viz_mask(self, imsize=(8, 8)):
        plt.figure(figsize=imsize)
        plt.imshow(self.mask, cmap='Greys_r')
        plt.axis('off')
        plt.show()
        
    def viz_image(self, imsize=(8, 8)):
        plt.figure(figsize=imsize)
        plt.imshow(self.image, cmap='Greys_r')
        plt.axis('off')
        plt.show()

    def score_crops(self, classifier):
        scores = classifier.forward_cropset(self.crops)
        self.crops.update_scores(scores)

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

    
