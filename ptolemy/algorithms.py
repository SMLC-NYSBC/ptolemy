# Contain box angle finding, lattice fitting algos, maybe some others
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import label
from PoissonMixture import PoissonMixture
import geometry as geom
from scipy.optimize import minimize_scalar
import math
from models import BasicUNet, Wrapper
import torch


def flood_segments(mask, search_size):
    """
    Generates and combines segments from mask pixel-wise, with all positive pixels within search_size/2 of each other being combined to the same segment. 
    """
    orig = np.copy(mask)
    search_mask = np.ones((search_size, search_size))
    expand = convolve2d(orig, search_mask, mode='same') > 0
    objects, n_objects = label(expand, structure=np.ones((3, 3)))
    segments = orig * objects
    return segments, n_objects

def best_rot_angle(polygons, img_shape):
    origin = img_shape // 2
    def get_optim_fun(polygons):
        def f(angle):
            total = 0
            for poly in polygons:
                rotated_poly = poly.rotate_around_point(math.radians(angle), origin, origin)
                total += (rotated_poly.ymax() - rotated_poly.ymin()) * (rotated_poly.xmax() - rotated_poly.xmin())
            return total
        return f

    return minimize_scalar(get_optim_fun(polygons), bounds=(0, 90), method='bounded')['x']

class PMM_Segmenter:
    def __init__(self, search_size=6, remove_area_lt=100):
        self.search_size = 6
        self.remove_area_lt = 100
    
    def forward(self, image):
        model = PoissonMixture()
        model.fit(image.astype(int), verbose=False)
        mask = model.mask
        
        segments, num_regions = geom.flood_segments(mask, self.search_size)
        polygons = geom.segments_to_polygons(segments)
        polygons = [poly for poly in polygons if poly.area() > self.remove_area_lt]
        opt_rot_degrees = best_rot_angle(polygons, image.shape)
        boxes, rotated_boxes, rotated_image = geom.get_boxes_from_angle(image, polygons, opt_rot_degrees)
        return boxes, rotated_boxes, rotated_image, opt_rot_degrees, mask


class LowMag_Process_Crops:
    def __init__(self, normalize=True, width=240):
        self.normalize = normalize
        self.width = width

    def intensity_mean_std(self, exposure):
        intensities = (exposure.image * exposure.mask).flatten()
        intensities = intensities[intensities != 0]
        return np.mean(intensities), np.std(intensities)
    
    def forward(self, exposure, crops):
        crops.reshape(width=self.width)
        if self.normalize:
            mean, std = self.intensity_mean_std(self, exposure)
        crops.normalize(mean, std)
        return crops


class UNet_Segmenter:
    def __init__(self, channels, layers, model_path, threshold=0.0001):
        model = BasicUNet(channels, layers)
        model.load_state_dict(torch.load(model_path))
        self.model = Wrapper(model)
    
    def forward(self, image):
        results = self.model.forward_single(image)
        

