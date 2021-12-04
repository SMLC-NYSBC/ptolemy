# Combine image w/ MRC
import numpy as np
from PIL import Image
import mrc
from algorithms import flood_segments
import copy
from CropSet import CropSet2D


def load_mrc(path):
    with open(path, 'rb') as f:
        content = f.read()
    im,_,_ = mrc.parse(content)
    return im


class Low_Mag_Exposure:
    def __init__(self, image, scale=1, operator_selections=None):
        # image is assumed to be unrotated, boxes also unrotated

        if np.sum(image < 0) != 0:
            image = np.copy(image)
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

    def segment(self, segment_alg):
        self.boxes, self.rotated_boxes, self.rotated_image, self.rot_ang_deg, self.mask = segment_alg.forward(self.image)
        self.box_areas = [box.area() for box in self.boxes]
        self.segment_alg = segment_alg
        # return copy.deepcopy(self.boxes), copy.deepcopy(self.rotated_boxes), copy.deepcopy(self.rotated_image), copy.deepcopy(self.rot_ang_deg)

    def viz_boxes(self, rotated=False, selections=False):
        # plt.plot image and boxes 
        raise NotImplementedError

    def viz_boxes_and_scores(self, rotated=False, selections=False):
        # plt.plot image and boxes with nice viz
        raise NotImplementedError

    def get_crops(self, postprocess_alg=None):
        if not hasattr(self, 'rotated_boxes'):
            raise ValueError
        crops = []
        for box in self.rotated_boxes:
            segmented_box = self.rotated_image[max(box.ymin(), 0): min(box.ymax(), self.rotated_image.shape[0]) ,
                                               max(box.xmin(), 0): min(box.xmax(), self.rotated_image.shape[1]) ]
            crops.append(segmented_box)

        crops = CropSet2D(crops, self.boxes, self.rotated_boxes)

        if postprocess_alg is not None:
            crops = postprocess_alg.forward(self, crops)
        
        self.crops = crops
        return crops

    def score_crops(self, classifier):
        scores = classifier.forward_cropset(self.crops)
        self.crops.update_scores(scores)

    