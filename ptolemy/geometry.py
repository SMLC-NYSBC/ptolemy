# Contain simple geometry functions
from ptolemy.PointSet import PointSet2D
import numpy as np
from scipy.spatial import ConvexHull
from scipy.ndimage import rotate
import math



############### POLYGONS #################
def convex_hull(points):
    hull = ConvexHull(points)
    vertices = hull.vertices
    points = points[vertices]
    return PointSet2D(points[:, 0], points[:, 1])


def segments_to_polygons(segments):
    """
    For each segment, returns the polygon surounding it by solving for the convex hull.
    """

    n = int(np.max(segments))
    polygons = []

    for i in range(1, n+1):
        # get the points for this region
        I,J = np.where(segments == i)
        # define every pixel as a box defined by four corners
        corners = [
            np.stack([I, J], axis=1),
            np.stack([I+1, J], axis=1),
            np.stack([I+1, J+1], axis=1),
            np.stack([I, J+1], axis=1),
        ]
        corners = np.concatenate(corners, axis=0)
        vertices = convex_hull(corners)
        polygons.append(vertices)

    return polygons


def get_boxes_from_angle(image, polygons, degrees):
    min_img = min(image.shape[0], image.shape[1])
    rotated_image = rotate(image, degrees)
    min_rotated= min(rotated_image.shape[0], rotated_image.shape[1])
    
    # original_origin = [image.shape[0] // 2, image.shape[1] // 2]
    # rotated_origin = [rotated_image.shape[0] // 2, rotated_image.shape[1] // 2] 
    
    original_origin = [min_img // 2, min_img // 2] 
    rotated_origin = [min_rotated // 2, min_rotated // 2]
    
    rotated_boxes = []
    boxes = []
    
    for polygon in polygons:
        polygon = polygon.rotate_around_point(math.radians(degrees), original_origin, rotated_origin)
        
        xmin = polygon.xmin()
        xmax = polygon.xmax()
        ymin = polygon.ymin()
        ymax = polygon.ymax()

#         box = PointSet2D([ymin, ymax, ymax, ymin], [xmin, xmin, xmax, xmax])
        box = PointSet2D([xmin, xmin, xmax, xmax], [ymin, ymax, ymax, ymin])
        rotated_boxes.append(box)
        
        rotated_box_back = box.rotate_around_point(math.radians(degrees), rotated_origin, original_origin)
        boxes.append(rotated_box_back)
        
    return boxes, rotated_boxes, rotated_image

def get_centroids_for_polygons(polygons):
    centroids = np.stack([poly.center_of_mass() for poly in polygons], axis=0)
    centroids = PointSet2D(centroids[:, 0], centroids[:, 1])
    return centroids

def pad_crops(crops, width):
    new_crops = []
    for box in crops:
        if box.shape[0] > width:
            xcenter = box.shape[0] // 2
            xmin, xmax = xcenter - width//2, xcenter + width//2
            box = box[xmin:xmax, :]
        else:
            left_right_concat = np.zeros(( (width-box.shape[0])//2, box.shape[1]))
            box = np.concatenate((left_right_concat, box, left_right_concat), axis=0)

        if box.shape[1] > width:
            ycenter = box.shape[1] // 2
            ymin, ymax = ycenter - width//2, ycenter + width//2
            box = box[:, ymin:ymax]
        else:
            top_bottom_concat = np.zeros((box.shape[0], (width - box.shape[1])//2))
            box = np.concatenate((top_bottom_concat, box, top_bottom_concat), axis=1)
        
        if box.shape[0] != width:
            box = np.concatenate((np.zeros((1, box.shape[1])), box), axis=0)
        if box.shape[1] != width:
            box = np.concatenate((np.zeros((box.shape[0], 1)), box), axis=1)
        new_crops.append(box)

    return new_crops

def normalize_crops_using_mask(crops, lm_image, mask):
    intensities = (lm_image * mask).flatten()
    intensities = intensities[intensities != 0]
    mean, std = intensities.mean(), intensities.std()

    normalized_crops = []
    for crop in crops:
        normalized_crops.append((crop - mean) / std)

    return normalized_crops