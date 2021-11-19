# Combine image w/ MRC
import numpy as np
from PIL import Image
import ptolemy.mrc as mrc
from algorithms import flood_segments


def load_mrc(path):
    with open(path, 'rb') as f:
        content = f.read()
    im,_,_ = mrc.parse(content)
    return im


class Exposure:
    def __init__(self, image, scale=1, operator_selections=None):
        # image is assumed to be unrotated, boxes also unrotated

        if np.sum(image < 0) != 0:
            image = np.copy(image)
            image[image < 0] = 0

        self.image = image
        self.scale = scale

        self.size = image.shape
        self.image_scaled = image
        if scale > 1:
            n = int(np.round(image.shape[0]/scale))
            m = int(np.round(image.shape[1]/scale))
            self.size = (n,m)
            self.image_scaled = downsample(image, self.size)
        
        self.operator_selections = operator_selections

    def segment(self, segment_alg):
        self.boxes, self.rotated_boxes, self.rot_ang_deg = segment_alg(self.image)
        return self.boxes, self.rotated_boxes, self.rot_ang_deg

    def viz_boxes(self, rotated=False, selections=False):
        # plt.plot image and boxes 
        raise NotImplementedError

    def viz_boxes_and_scores(self, rotated=False, selections=False):
        # plt.plot image and boxes with nice viz
        raise NotImplementedError

    def make_crops(self):
        # make a cropset out of the boxes
        raise NotImplementedError
    

    