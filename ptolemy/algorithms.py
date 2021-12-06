# Contain box angle finding, lattice fitting algos, maybe some others
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import label
from PointSet import PointSet2D
from PoissonMixture import PoissonMixture
import geometry as geom
from scipy.optimize import minimize_scalar
import math
from models import BasicUNet, Wrapper
import torch
from scipy.spatial.distance import cdist, euclidean, pdist, squareform, 



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

def grid_from_centroids(centroids, mask, gp_padding=15, fn_weight=10):
    # need to adjust to use pointset2D
    distmat = squareform(pdist(centroids.as_matrix()))
    best_error = np.inf

    for i in range(len(distmat)):
        row = distmat[i]
        topk = np.argpartition(row, np.min(len(row), 6))[:6]
        for j in topk:
            if i == j:
                continue
            gen_gps, angle = generate_gp(centroids, i, j, mask.shape)
            err = gp_mask_error(gen_gps, mask, gp_padding, fn_weight)
            if err < best_error:
#                 print(err)
#                 plt.imshow(gen_mask, cmap='Greys_r')
#                 plt.show()
                best_gps = gen_gps
                best_angle = angle
                best_error = err
    # compute error
    return best_gps, best_angle

def generate_gp(centroids, i, j, shape):
    anchor_i = [centroids.y[i], centroids.x[i]]
    anchor_j = [centroids.y[j], centroids.x[j]]

    angle_rad = np.arctan((anchor_j[0] - anchor_i[0])/(anchor_j[1] - anchor_i[1]))
    angle_deg = math.degrees(angle_rad)

    anchor_line, orth_dx, orth_dy = get_anchor_line(anchor_i, anchor_j, shape)
    grid_pts = []
    for pt_y, pt_x in zip(anchor_line.y, anchor_line.x):
        grid_pts.append(get_perp_line(pt_y, pt_x, orth_dx, orth_dy))

    grid_pts = PointSet2D.concatenate(grid_pts)

    grid_pts = grid_pts.bound_pts_imshape(shape)
    return grid_pts, angle_deg

def get_anchor_line(anchor_i, anchor_j):
    y_i, x_i = anchor_i[0], anchor_i[1]
    y_j, x_j = anchor_j[0], anchor_j[1]

    dx = x_j - x_i
    dy = y_j - y_i

    x_l1 = np.arange(-100, 100)
    y_l1 = np.arange(-100, 100)
    x_l1 = (x_l1 * dx) + x_i
    y_l1 = (y_l1 * dy) + y_i

    anchor_line = PointSet2D(y_l1, x_l1)

    angle = np.arctan((y_j - y_i)/(x_j - x_i))
    orth_angle = angle + (np.pi/2)
    d = np.sqrt((x_j - x_i)**2 + (y_j - y_i)**2)
    orth_dx = d * np.cos(orth_angle)
    orth_dy = d * np.sin(orth_angle)

    return anchor_line, orth_dx, orth_dy

def get_perp_line(pt_y, pt_x, orth_dx, orth_dy):
    x_l = np.arange(-100, 100)
    y_l = np.arange(-100, 100)
    x_l = (x_l * orth_dx) + pt_x
    y_l = (y_l * orth_dy) + pt_y

    perp_line = PointSet2D(y_l, x_l)
    return perp_line

def gp_mask_error(gen_gps, target_mask, gp_padding, fn_weight):
    gp_mask = np.zeros(target_mask.shape)
    for gp_y, gp_x in zip(gen_gps.y, gen_gps.x):
        gp_mask[gp_y-gp_padding:gp_y+gp_padding, gp_x-gp_padding:gp_x+gp_padding] = 1
    
    fn = gp_mask - target_mask
    fn = fn[fn < 0]

    fp = gp_mask - target_mask
    fp = fp[fp > 0]
    
    return (fn_weight * np.sum(fn ** 2)) + np.sum(fp ** 2)


class PMM_Segmenter:
    def __init__(self):
        self.model = PoissonMixture()
    
    def forward(self, image):
        self.model.fit(image.astype(int), verbose=False)
        mask = self.model.mask
        return mask

class Low_Mag_Process_Mask:
    def __init__(self, search_size=6, remove_area_lt=100):
        self.search_size = search_size
        self.remove_area_lt = remove_area_lt

    def forward(self, mask):
        segments, _ = geom.flood_segments(mask, self.search_size)
        polygons = geom.segments_to_polygons(segments)
        polygons = [poly for poly in polygons if poly.area() > self.remove_area_lt]
        opt_rot_degrees = best_rot_angle(polygons, mask.shape)
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

class MedMag_Process_Mask:
    def __init__(self, seg_search_size=6, gp_padding=15, fn_weight=10): #todo):
        self.seg_search_size = seg_search_size
        self.gp_padding = gp_padding
        self.fn_weight = fn_weight

    def forward(self, mask):
        segments, _ = flood_segments(mask, self.seg_search_size)
        polygons = geom.segments_to_polygons(segments)
        centroids = geom.centroids_for_polygons(polygons)
        best_gps, angle = grid_from_centroids(centroids, mask, self.gp_padding, self.fn_weight)



        



