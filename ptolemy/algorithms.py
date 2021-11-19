# Contain box angle finding, lattice fitting algos, maybe some others
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import label
from PoissonMixture import PoissonMixture

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

class PMM_SquareSeg:
    def __init__(self, flood_segments_search_size=6, remove_unscaled_area_lt=100):
