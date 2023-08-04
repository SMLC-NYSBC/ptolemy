import os
import json
import sys
import math
from math import cos, sin

path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, path)

import numpy as np
import pandas as pd
import torch
import torchvision
from scipy.stats import skew, kurtosis
from scipy.ndimage import rotate, gaussian_filter
from scipy.special import expit
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks, rescale
from skimage.draw import disk

from PoissonMixture import PoissonMixture
from algorithms import flood_segments, grid_from_centroids, best_rot_angle
import geometry as geom
from PointSet import PointSet2D
import models


class Ptolemy:
    """
    Base class for static Ptolemy models. semi-Stateless. Use for prior model scores, feature extraction and hole/square detection. 

    Main functions are 
        load_models
        process_lm_image
        process_mm_image

    But you can also optionally only load/run parts of this class such as the constitutent functions in process_lm_image and process_mm_image
    
    Note that noice_hole_intensity is an important parameter and should be set for each session. If it not set, the model will default to using the 1.2*(90th percentile pixel intensity) as the no-ice-pixel-intensity for each image. 
    """

    def __init__(self, config='default'):
        if config == 'default':
            config_path = os.path.dirname(os.path.realpath(__file__)) + '/default_config_cpu.json'
        else:
            config_path = config
        self.load_config_and_models(config_path)


    def load_models(self):
        self.lm_classification_model = models.LowMag_64x5_2ep()
        self.lm_classification_model.load_state_dict(torch.load(self.settings['lm_prior_classication_model_path']))
        self.lm_classification_model.to(self.device)

        self.mm_segmenter = models.BasicUNet(64, 9)
        self.mm_segmenter.load_state_dict(torch.load(self.settings['mm_segmentation_model_path']))
        self.mm_segmenter.to(self.device)

        self.mm_feature_extraction_model = models.Hole_Classifier_Multitask(7, 256)
        self.mm_feature_extraction_model.load_state_dict(torch.load(self.settings['mm_feature_extraction_model_path']))
        self.mm_feature_extraction_model.to(self.device)

        self.mm_prior_model = models.CNNModel()
        self.mm_prior_model.load_state_dict(torch.load(self.settings['mm_prior_model_path']))
        self.mm_prior_model.to(self.device)


    def update_noice_hole_intensity(self, intensity):
        self.settings['noice_hole_intensity'] = intensity
        

    def load_config_and_models(self, config_path):
        self.settings = json.load(open(config_path, 'r'))
        self.device = self.settings['device']
        if self.device == 'cuda':
            os.environ['CUDA_VISIBLE_DEVICES'] = self.settings['CUDA_VISIBLE_DEVICES']
        self.load_models()


    def process_lm_image(self, lm_image):
        raw_crops, preprocessed_crops, centers, vertices, areas, mean_intensities = self.get_lm_crops(lm_image) # TODO return boxes
        features = self.get_lm_features(raw_crops)
        prior_scores = self.get_lm_prior_scores(preprocessed_crops, features)
        return raw_crops, centers, vertices, areas, mean_intensities, features, prior_scores


    def process_mm_image(self, mm_image):
        crops, masked_crops, centers, radii, boxes = self.get_mm_crops(mm_image)
        batch = self.mm_crops_to_preprocessed_batch(masked_crops, radii)
        features = self.get_mm_features(batch)
        prior_scores = self.get_mm_prior_scores(batch, features)
        return crops, centers, boxes, radii, features, prior_scores


    ############# Low Mag Processing Functions ############

    def get_lm_crops(self, lm_image):
        mask, segments = self._segment_lm(lm_image)
        polygons = geom.segments_to_polygons(segments)
        
        polygons, segment_indices = self._filter_polygon_areas(polygons)

        opt_rot_degrees = best_rot_angle(polygons, lm_image.shape)
        boxes, rotated_boxes, rotated_image = geom.get_boxes_from_angle(lm_image, polygons, opt_rot_degrees)

        crops = self._crops_from_vertices(rotated_boxes, rotated_image) # could probably move to geom
        crops, boxes, segment_indices = self._filter_crop_size(crops, boxes, segment_indices)

        preprocessed_crops = geom.pad_crops(crops, self.settings['lm_crop_width'])
        preprocessed_crops = geom.normalize_crops_using_mask(preprocessed_crops, lm_image, mask)

        mean_intensities = [np.mean(lm_image[(segments == index)]) for index in segment_indices] # TODO eventually we might want to try collecting features from the segments not the crops

        areas = [box.area() for box in boxes]
        vertices = [box.as_matrix_y().tolist() for box in boxes]
        center_coords = PointSet2D.concatenate([PointSet2D([int(np.mean(box.y))], [int(np.mean(box.x))]) for box in boxes])
        center_coords = np.round(center_coords.as_matrix_y()).astype(int).tolist()
        
        return crops, preprocessed_crops, center_coords, vertices, areas, mean_intensities


    def _segment_lm(self, lm_image):
        pmm = PoissonMixture()
        pmm.fit(lm_image.astype(int), verbose=False)
        mask = pmm.mask

        segments, _ = flood_segments(mask, self.settings['lm_segment_search_size'])
        return mask, segments


    def _filter_polygon_areas(self, polygons):
        polys, indicies = [], []
        for i, poly in enumerate(polygons, 1):
            if poly.area() > self.settings['lm_remove_area_lt']:
                polys.append(poly)
                indicies.append(i)
        
        return polys, indicies


    def _crops_from_vertices(self, rotated_boxes, rotated_image):
        crops = []
        for rotated_box in rotated_boxes:
            crop = rotated_image[int(max(rotated_box.xmin(), 0)): int(min(rotated_box.xmax(), rotated_image.shape[0])) ,
                                 int(max(rotated_box.ymin(), 0)): int(min(rotated_box.ymax(), rotated_image.shape[1])) ]
            # TODO possibly consider minimum box size
            crops.append(crop)
        
        return crops


    def _filter_crop_size(self, crops, boxes, segment_indices):
        ret_crops, ret_boxes, ret_segment_indices = [], [], []
        for crop, box, index in zip(crops, boxes, segment_indices):
            if crop.shape[0] < self.settings['lm_square_min_width'] or crop.shape[1] < self.settings['lm_square_min_width']:
                continue
            ret_crops.append(crop)
            ret_boxes.append(box)
            ret_segment_indices.append(index)

        return ret_crops, ret_boxes, ret_segment_indices


    def get_lm_features(self, crops):
        """
        Getting lm features from aggregate statistics of crop pixels

        TODO in the future maybe use the pixels rather than the crops
        """
        feats = []
        for crop in crops:
            feats.append(np.array([
                np.mean(crop),
                np.max(crop),
                # np.min(crop),
                np.var(crop),
                np.float32(crop.shape[0] * crop.shape[1]),
                # np.float32(kurtosis(crop, axis=None)),
                # np.float32(skew(crop, axis=None))
            ]))
        
        return feats


    def get_lm_prior_scores(self, preprocessed_crops, features):
        scores = self.lm_classification_model.score_crops(preprocessed_crops)
        return scores

    
    ############# Med Mag Processing Functions ############

    def get_mm_crops(self, mm_image):
        preprocessed_image = self._preprocess_mm_image(mm_image)

        mask = self._get_mm_mask(preprocessed_image)
        best_gps, angle, distance = self._find_mm_grid(mask)

        rotated_image = rotate(mm_image, -angle)

        boxes, rotated_boxes = self._extract_mm_rotated_boxes(mm_image, rotated_image, angle, best_gps, distance)
        crops, boxes, rotated_boxes = self._mm_extract_crops(boxes, rotated_boxes, rotated_image)

        centers = PointSet2D.concatenate([PointSet2D([int(np.mean(box.y))], [int(np.mean(box.x))]) for box in boxes])
        centers = np.round(centers.as_matrix_y()).astype(int).tolist()

        radii = self._mm_hole_radii(crops)

        boxes = [box.as_matrix_y() for box in boxes]
        
        if self.settings['mm_filter_cutoff_holes']:
            # TODO add option to filter this more tolerantly
            crops, centers, radii, boxes = self._filter_cutoff_holes(mm_image, crops, centers, radii, boxes)
            
        if len(crops) > self.settings['mm_refine_centers_if_more_holes_than']:
            centers_relative_to_crops, radii = self._refine_center_and_radii(crops, radii)
            centers = self._apply_center_refinement(centers_relative_to_crops, angle, boxes)
        else:
            centers_relative_to_crops = [[crop.shape[0] // 2, crop.shape[1] // 2] for crop in crops]

        masked_crops = self._make_mask_crops(crops, centers_relative_to_crops, radii)
        
        return crops, masked_crops, centers, radii, boxes


    def _get_mm_mask(self, mm_image):
        mask = self.mm_segmenter.get_mask(mm_image)
        mask = expit(mask)
        mask[mask > self.settings['mm_mask_threshold']] = 1
        mask[mask <= self.settings['mm_mask_threshold']] = 0
        return mask[:mm_image.shape[0], :mm_image.shape[1]]


    def _find_mm_grid(self, mask):
        segments, _ = flood_segments(mask, self.settings['mm_segment_search_size'])
        polygons = geom.segments_to_polygons(segments)
        if len(polygons) < 2:
            raise ValueError('Fewer than two holes detected by U-Net in med mag image')
        centroids = geom.get_centroids_for_polygons(polygons)
        best_gps, angle, distance = grid_from_centroids(centroids, mask, self.settings['mm_gridpoint_padding'], self.settings['mm_fn_weight'])
        best_gps = best_gps.bound_pts_imshape(mask.shape, tolerance=self.settings['mm_gridpoint_edge_tolerance'])

        return best_gps, angle, distance


    def _extract_mm_rotated_boxes(self, mm_image, rotated_image, angle, best_gps, distance):
        init_origin = [mm_image.shape[0] // 2, mm_image.shape[1] // 2]
        rotated_origin = [rotated_image.shape[0] // 2, rotated_image.shape[1] // 2]
        rotated_gps = best_gps.rotate_around_point(math.radians(angle), init_origin, rotated_origin)
        
        boxes = []
        rotated_boxes = []
        crop_distance = distance // 2 - self.settings['mm_crop_side_pixels']
        for center_y, center_x in zip(rotated_gps.y, rotated_gps.x):
            rotated_box = PointSet2D([center_y - crop_distance, center_y + crop_distance],
                                       [center_x - crop_distance, center_x + crop_distance]).get_bounding_box()
            rotated_boxes.append(rotated_box)
            boxes.append(rotated_box.rotate_around_point(-math.radians(angle), rotated_origin, init_origin))

        return boxes, rotated_boxes


    def _preprocess_mm_image(self, mm_image):
        """
        Currently we only do normalization and padding to a dimension allowed by the u-net
        In the future we may want to do a high-pass filter
        As well as correct any tilt
        """
        allowed_dims = [self.settings['mm_dim_mult_of'] * x for x in range(1, 5)]

        preprocessed_image = (mm_image - mm_image.mean()) / mm_image.std()

        mean = mm_image.mean()
        
        if preprocessed_image.shape[0] not in allowed_dims:
            if preprocessed_image.shape[0] > allowed_dims[-1]:
                raise ValueError(f'Medium mag image with size {preprocessed_image.shape} too large. Please downsample to max dimension {allowed_dims[-1]}')
            for min_, max_ in zip(allowed_dims[:-1], allowed_dims[1:]):
                if preprocessed_image.shape[0] > min_ and preprocessed_image.shape[0] < max_:
                    addon = np.zeros((max_ - preprocessed_image.shape[0], preprocessed_image.shape[1]))
                    addon = addon + mean
                    preprocessed_image = np.concatenate((preprocessed_image, addon), axis=0)
        
        if preprocessed_image.shape[1] not in allowed_dims:
            if preprocessed_image.shape[1] > allowed_dims[-1]:
                raise ValueError(f'Medium mag image with size {preprocessed_image.shape} too large. Please downsample to max dimension {allowed_dims[-1]}')
            for min_, max_ in zip(allowed_dims[:-1], allowed_dims[1:]):
                if preprocessed_image.shape[1] > min_ and preprocessed_image.shape[1] < max_:
                    addon = np.zeros((preprocessed_image.shape[0], max_ - preprocessed_image.shape[1]))
                    addon = addon + mean
                    preprocessed_image = np.concatenate((preprocessed_image, addon), axis=1)

        return preprocessed_image


    def _mm_consensus_shape(self, crops):
        shapes = [crop.shape for crop in crops]
        shapes = pd.Series(shapes).value_counts()
        if len(shapes) == 1 and shapes.iloc[0] > 2:
            return shapes.index[0]
        if (shapes.iloc[0] == shapes.iloc[1]) or shapes.iloc[0] <= 2:
            raise ValueError('Unable to find a consensus shape for medium mag crops')
            
        return shapes.index[0]


    def _mm_hole_radii(self, crops):
        """
        This method should return a radii for each crop. 
        Currently we find a consensus radius with this method but for futureproofing, 
        it should return a list of length(crops)
        """
        consensus_shape = self._mm_consensus_shape(crops)
        init = np.zeros(consensus_shape)
        for crop in crops:
            if crop.shape == consensus_shape and np.sum(np.isnan(crop)) == 0 and not (crop == 0).any():
                init = init + crop
        
        if init.sum() == 0:
            raise ValueError('No good hole crops to find hole radius found')

        init = gaussian_filter(init, 5)
        edges = canny(init)
        radii = np.arange(init.shape[0] // 5, init.shape[0] // 2, 1)
        hough_res = hough_circle(edges, radii)
        _, cx, cy, radii = hough_circle_peaks(hough_res, radii, total_num_peaks=1)
        if len(cx) == 0:
            raise ValueError('No circles detected in stacked crops')

        return [radii[0]] * len(crops)

    
    def _mm_extract_crops(self, boxes, rotated_boxes, rotated_image):
        ret_crops = []
        ret_boxes = []
        ret_rotated_boxes = []

        for box, rotated_box in zip(boxes, rotated_boxes):
            segmented_box = rotated_image[int(max(rotated_box.xmin(), 0)): int(min(rotated_box.xmax(), rotated_image.shape[0])) ,
                                          int(max(rotated_box.ymin(), 0)): int(min(rotated_box.ymax(), rotated_image.shape[1])) ]
            if self.settings['mm_min_length'] and (segmented_box.shape[0] < self.settings['mm_min_length'] or segmented_box.shape[1] < self.settings['mm_min_length']):
                continue
            ret_crops.append(segmented_box)
            ret_boxes.append(box)
            ret_rotated_boxes.append(rotated_box)

        return ret_crops, ret_boxes, ret_rotated_boxes


    def _filter_cutoff_holes(self, image, crops, centers, radii, boxes):
        image_x, image_y = image.shape

        rcrops, rcenters, rradii, rboxes = [], [], [], []
        for crop, center, radius, box in zip(crops, centers, radii, boxes):
            if center[0] - radius < 0 or center[0] + radius > image_y:
                continue
            elif center[1] - radius < 0 or center[1] + radius > image_x:
                continue

            rcrops.append(crop)
            rcenters.append(center)
            rradii.append(radius)
            rboxes.append(box)

        return rcrops, rcenters, rradii, rboxes
    
    
    def _refine_center_and_radii(self, crops, radii):
        centers_relative_to_crops, new_radii = [], []

        for crop, radius in zip(crops, radii):
            init = gaussian_filter(crop, 5)
            edges = canny(init)
            search_radii = np.arange(radius - self.settings['mm_refinement_radius_search_width'],
                                     radius + self.settings['mm_refinement_radius_search_width'],
                                     self.settings['mm_refinement_radius_search_granularity'])
            
            hough_res = hough_circle(edges, search_radii)
            _, cx, cy, radii = hough_circle_peaks(hough_res, search_radii, total_num_peaks=1)
            if len(cx) == 0:
                print('no new circle found, returning original center and radius')
                cx = crop.shape[0] // 2
                cy = crop.shape[1] // 2
                radius = radius
            else:
                cx = cx[0]
                cy = cy[0]
                radius = radii[0]

            centers_relative_to_crops.append([cx, cy])
            new_radii.append(radius)
        
        return centers_relative_to_crops, new_radii


    def _apply_center_refinement(self, centers_relative_to_crops, angle, bounding_boxes):
        centers_relative_to_crops = np.array(centers_relative_to_crops)
        theta = np.deg2rad(angle)
        rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        
        rotated_centers_relative_to_crops = np.dot(centers_relative_to_crops, rot)
        centers = [(rel_center[0] + bbox[0][0], rel_center[1] + bbox[0][1]) for 
                    rel_center, bbox in zip(rotated_centers_relative_to_crops, bounding_boxes)]
        return centers


    def _make_mask_crops(self, crops, centers_relative_to_crops, radii):
        masked_crops = []

        for crop, center, radius in zip(crops, centers_relative_to_crops, radii):
            rr, cc = disk(center, radius, shape=crop.shape)
            mask = np.zeros(crop.shape)
            safe_cc = cc[(cc > 0) & (cc < crop.shape[0])]
            safe_rr = rr[(cc > 0) & (cc < crop.shape[0])]
            safe_cc = safe_cc[(safe_rr > 0) & (safe_rr < crop.shape[1])]
            safe_rr = safe_rr[(safe_rr > 0) & (safe_rr < crop.shape[1])]
            mask[safe_cc, safe_rr] = 1
            
            masked = crop * mask
            masked_crops.append(masked)

        return masked_crops


    def get_mm_features(self, batch):
        # applied transformations:
        # no normalization
        # edge tolerance = 100
        # get radius
        # resize to target radius of 70
        # cutoff filter - let's not do this actually
        # normalize by session intensity which must be provided by shooting at no ice hole
        # Feed it into the multitask model and get features
        # Done
        features = self.mm_feature_extraction_model.extract_features(batch)

        return features


    def _mm_resize_to_radius(self, crop, radius, target_radius):
        if radius == target_radius:
            return crop
        else:
            resize_fraction = target_radius / radius
            rescaled = rescale(crop, resize_fraction, anti_aliasing=True)
            return rescaled

    def get_mm_prior_scores(self, batch, features):
        # run the contaminant model
        # Our just runs on crops
        return self.mm_prior_model.score_batch(batch)

    
    def mm_crops_to_preprocessed_batch(self, crops, radii):
        rescaled_crops = [self._mm_resize_to_radius(crop, radius, self.settings['target_hole_radius']) for crop, radius in zip(crops, radii)]
        if 'noice_hole_intensity' not in self.settings.keys() or self.settings['noice_hole_intensity'] == -1:
            print('Warning: no-ice hole pixel intensity not set. Defaulting to 1.2 * (90th percentile pixel intensity) as the normalization factor')
            all_pixels_flat = []
            for crop in crops:
                all_pixels_flat.append(crop.flatten())
            all_pixels_flat = np.concatenate(all_pixels_flat)
            norm_factor = 1.2 * np.nanpercentile(all_pixels_flat, 90)
            rescaled_crops = [crop / norm_factor for crop in rescaled_crops]
        
        else:
            rescaled_crops = [crop / self.settings['noice_hole_intensity'] for crop in rescaled_crops]

        batch = self._crops_to_fixed_size_batch(rescaled_crops, self.settings['mm_input_crop_size'])
        return batch


    def _crops_to_fixed_size_batch(self, crops, size):
        batch = np.zeros((len(crops), 1, size, size), dtype='float32')
        for i, crop in enumerate(crops):
            if crop.shape[0] > size or crop.shape[1] > size:
                center = np.array([crop.shape[0] // 2, crop.shape[1] // 2])
                crop = crop[center[0] - (size//2):center[0] + (size//2), center[1] - (size//2):center[1] + (size//2)]
                
            start_ind0 = (size - crop.shape[0]) // 2
            end_ind0 = start_ind0 + crop.shape[0]
            start_ind1 = (size - crop.shape[1]) // 2
            end_ind1 = start_ind1 + crop.shape[1]
            
            batch[i, 0, start_ind0:end_ind0, start_ind1:end_ind1] = crop

        return batch