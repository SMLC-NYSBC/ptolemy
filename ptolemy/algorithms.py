# Contain box angle finding, lattice fitting algos, maybe some others
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import label, rotate
from ptolemy.PointSet import PointSet2D
from ptolemy.PoissonMixture import PoissonMixture
import ptolemy.geometry as geom
from scipy.optimize import minimize_scalar
import math
from ptolemy.models import BasicUNet, Wrapper
import torch
from scipy.spatial.distance import cdist, euclidean, pdist, squareform 
from scipy.special import expit
import matplotlib.pyplot as plt
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
import pandas as pd
from scipy.ndimage import gaussian_filter

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
    origin = [img_shape[0] // 2, img_shape[1] // 2]
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
    distmat = squareform(pdist(centroids.as_matrix_y()))
    best_error = np.inf

    for i in range(len(distmat)):
        row = distmat[i]
        topk = np.argpartition(row, min(len(row)-1, 6))[:min(len(row) - 1, 6)]
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
                best_distance = distmat[i, j]
    # compute error
    try:
        return best_gps, best_angle, best_distance
    except:
        raise BadMedMagError

def generate_gp(centroids, i, j, shape):
    anchor_i = [centroids.y[i], centroids.x[i]]
    anchor_j = [centroids.y[j], centroids.x[j]]

    angle_rad = np.arctan((anchor_j[0] - anchor_i[0])/(anchor_j[1] - anchor_i[1]))
    angle_deg = math.degrees(angle_rad)

    anchor_line, orth_dx, orth_dy = get_anchor_line(anchor_i, anchor_j)
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
    for gp_y, gp_x in zip(gen_gps.y.astype(int), gen_gps.x.astype(int)):
        gp_mask[gp_x-gp_padding:gp_x+gp_padding, gp_y-gp_padding:gp_y+gp_padding] = 1
    
    fn = gp_mask - target_mask
    fn = fn[fn < 0]

    fp = gp_mask - target_mask
    fp = fp[fp > 0]
    
    return (fn_weight * np.sum(fn ** 2)) + np.sum(fp ** 2)


class PMM_Segmenter:    
    def forward(self, image):
        self.model = PoissonMixture()
        self.model.fit(image.astype(int), verbose=False)
        mask = self.model.mask
        return mask

# class LowMag_Process_Mask:
#     def __init__(self, search_size=6, remove_area_lt=100):
#         self.search_size = search_size
#         self.remove_area_lt = remove_area_lt

#     def forward(self, mask, image):
#         segments, _ = flood_segments(mask, self.search_size)
#         polygons = geom.segments_to_polygons(segments)
#         mean_intensities = []
#         polygons_ = []
        
#         for i, poly in enumerate(polygons, 1):
#             if poly.area() > self.remove_area_lt:
#                 polygons_.append(poly)
#                 where = (segments == i)
#                 pixels = image[where]
#                 mean_intensities.append(np.mean(pixels))
        
#         polygons = polygons_
#         opt_rot_degrees = best_rot_angle(polygons, image.shape)
#         boxes, rotated_boxes, rotated_image = geom.get_boxes_from_angle(image, polygons, opt_rot_degrees)
#         return boxes, rotated_boxes, rotated_image, opt_rot_degrees, mean_intensities


# class LowMag_Process_Crops:
#     def __init__(self, normalize=True, width=240):
#         self.normalize = normalize
#         self.width = width

#     def intensity_mean_std(self, exposure):
#         intensities = (exposure.image * exposure.mask).flatten()
#         intensities = intensities[intensities != 0]
#         self.exposure_mean_intens = np.mean(intensities)
#         self.exposure_std_intens = np.std(intensities)
#         # return self.exposure_mean_intens, self.exposure_std_intens
    
#     def forward(self, exposure, crops):
#         crops.pad(width=self.width)
#         if self.normalize:
#             if not hasattr(self, 'exposure_mean_intens'):
#                 self.intensity_mean_std(exposure)
#         crops.normalize_constant(self.exposure_mean_intens, self.exposure_std_intens)
#         return crops


# class UNet_Segmenter:
#     def __init__(self, channels, layers, model_path, threshold=0.0001, normalize=True, cuda=False, dim_mult_of=512):
#         model = BasicUNet(channels, layers)
#         model.load_state_dict(torch.load(model_path))
#         self.model = Wrapper(model)
#         self.threshold = threshold
#         self.normalize = normalize
#         if cuda:
#             self.model.to_cuda()
#         self.dim_mult_of = dim_mult_of
#         self.allowed_dims = [dim_mult_of * x for x in range(1, 5)]
    
#     def forward(self, image):
#         if self.normalize:
#             image = (image - image.mean()) / image.std()
            
        
#         original_dim = image.shape
#         mean = image.mean()
        
#         if image.shape[0] < 1024:
#             addon = np.zeros((1024 - image.shape[0], image.shape[1]))
#             addon = addon + mean
#             image = np.concatenate((image, addon), axis=0)
            
#         elif image.shape[0] not in self.allowed_dims:
#             if image.shape[0] > self.allowed_dims[-1]:
#                 raise InputError
#             for min_, max_ in zip(self.allowed_dims[:-1], self.allowed_dims[1:]):
#                 if image.shape[0] > min_ and image.shape[0] < max_:
#                     addon = np.zeros((max_ - image.shape[0], image.shape[1]))
#                     addon = addon + mean
#                     image = np.concatenate((image, addon), axis=0)
            
            
#         if image.shape[1] < 1440:
#             addon = np.zeros((image.shape[0], 1440 - image.shape[1]))
#             addon = addon + mean
#             image = np.concatenate((image, addon), axis=1)
            
#         elif image.shape[1] not in self.allowed_dims:
#             if image.shape[1] > self.allowed_dims[-1]:
#                 raise InputError
#             for min_, max_ in zip(self.allowed_dims[:-1], self.allowed_dims[1:]):
#                 if image.shape[1] > min_ and image.shape[1] < max_:
#                     addon = np.zeros((image.shape[0], max_ - image.shape[1]))
#                     addon = addon + mean
#                     image = np.concatenate((image, addon), axis=1)
                    
#         mean = image.mean()
        
#         if image.shape[0] not in self.allowed_dims:
#             if image.shape[0] > self.allowed_dims[-1]:
#                 raise InputError
#             for min_, max_ in zip(self.allowed_dims[:-1], self.allowed_dims[1:]):
#                 if image.shape[0] > min_ and image.shape[0] < max_:
#                     addon = np.zeros((max_ - image.shape[0], image.shape[1]))
#                     addon = addon + mean
#                     image = np.concatenate((image, addon), axis=0)
        
#         if image.shape[1] not in self.allowed_dims:
#             if image.shape[1] > self.allowed_dims[-1]:
#                 raise InputError
#             for min_, max_ in zip(self.allowed_dims[:-1], self.allowed_dims[1:]):
#                 if image.shape[1] > min_ and image.shape[1] < max_:
#                     addon = np.zeros((image.shape[0], max_ - image.shape[1]))
#                     addon = addon + mean
#                     image = np.concatenate((image, addon), axis=1)
                    
#         results = self.model.forward_single(image)
#         results = expit(results)
#         results[results > self.threshold] = 1
#         results[results <= self.threshold] = 0
#         return results[:original_dim[0], :original_dim[1]]

# class MedMag_Process_Mask:
#     def __init__(self, seg_search_size=6, gp_padding=15, fn_weight=10, crop_sides=10, edge_tolerance=-10): #todo):
#         self.seg_search_size = seg_search_size
#         self.gp_padding = gp_padding
#         self.fn_weight = fn_weight
#         self.crop_sides = crop_sides
#         self.edge_tolerance = edge_tolerance
#         # TODO: Rationalize how large to make each crop

#     def forward(self, mask, image):
#         mean_intensities = None # we don't care about pixel intensities here
        
#         segments, _ = flood_segments(mask, self.seg_search_size)
#         polygons = geom.segments_to_polygons(segments)
#         if len(polygons) < 2:
#             raise BadMedMagError
#         centroids = geom.get_centroids_for_polygons(polygons)
#         best_gps, angle, distance = grid_from_centroids(centroids, mask, self.gp_padding, self.fn_weight)
        
#         best_gps = best_gps.bound_pts_imshape(image.shape, tolerance=self.edge_tolerance)
        
#         rotated_image = rotate(image, -angle)

#         init_origin = [image.shape[0] // 2, image.shape[1] // 2]
#         rotated_origin = [rotated_image.shape[0] // 2, rotated_image.shape[1] // 2]
#         rotated_gps = best_gps.rotate_around_point(math.radians(angle), init_origin, rotated_origin)
        
#         boxes = []
#         rotated_boxes = []
#         crop_distance = distance // 2 - self.crop_sides
#         for center_y, center_x in zip(rotated_gps.y, rotated_gps.x):
#             rotated_box = PointSet2D([center_y - crop_distance, center_y + crop_distance],
#                                        [center_x - crop_distance, center_x + crop_distance]).get_bounding_box()
#             rotated_boxes.append(rotated_box)
#             boxes.append(rotated_box.rotate_around_point(-math.radians(angle), rotated_origin, init_origin))

#         return boxes, rotated_boxes, rotated_image, angle, mean_intensities, best_gps

# class MedMag_Process_Crops:
#     def __init__(self, normalize=True):
#         self.normalize = normalize

#     def forward(self, exposure, cropset):
#         if len(cropset.crops) < 2:
#             raise BadMedMagError
#         if self.normalize:
#             cropset.normalize()
        
#         consensus_shape = self.get_consensus_shape(cropset.crops)
        
#         cat = []
#         for crop in cropset.crops:
#             if crop.shape == consensus_shape and np.sum(np.isnan(crop)) == 0 and not (crop == 0).any():
#                 cat.append(crop)
                
#         if len(cat) == 0:
#             raise BadMedMagError
                
#         start = cat[0]
#         for crop in cat[1:]:
#             start = start + crop
        
#         start = gaussian_filter(start, 5)
#         edges = canny(start)
#         radii = np.arange(start.shape[0] // 5, start.shape[0] // 2, 1)
#         hough_res = hough_circle(edges, radii)
#         _, cx, cy, radii = hough_circle_peaks(hough_res, radii, total_num_peaks=1)
#         if len(cx) == 0:
#             raise BadMedMagError
#         cx, cy, radii = cx[0], cy[0], radii[0]
        
#         exposure.hole_radii = radii
        
#         return cropset

#     def get_consensus_shape(self, crops):
#         shapes = [crop.shape for crop in crops]
#         shapes = pd.Series(shapes).value_counts()
#         if len(shapes) == 1 and shapes.iloc[0] > 2:
#             return shapes.index[0]
#         if (shapes.iloc[0] == shapes.iloc[1]) or shapes.iloc[0] <= 2:
#             raise BadMedMagError
            
#         return shapes.index[0]
            




        



