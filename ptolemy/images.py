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
    def __init__(self, image, scale=1, operator_selections=None, boxes=None):
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
        self.boxes = boxes

    def segment(self, search_size=6, remove_unscaled_area_lt=100):
        model = PoissonMixture()
        model.fit(self.image_scaled.astype(int))
        self._mask = model.mask

        self.segments, 


    

    