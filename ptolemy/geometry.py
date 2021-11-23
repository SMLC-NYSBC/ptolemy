# Contain simple geometry functions
from PointSet import PointSet2D
import numpy as np
from scipy.spatial import ConvexHull
from scipy.ndimage import rotate
import math



############### POLYGONS #################
def convex_hull(points):
    hull = ConvexHull(points)
    vertices = hull.vertices
    return PointSet2D(vertices[:, 0], vertices[:, 1])


def segments_to_polygons(segments):
    """
    For each segment, returns the polygon surounding it by solving for the convex hull.
    """

    n = np.max(segments)
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
    original_origin = image.shape // 2
    rotated_image = rotate(image, degrees)
    rotated_origin = rotated_image.shape // 2
    
    rotated_boxes = []
    boxes = []
    
    for polygon in polygons:
        polygon = polygon.rotate_around_point(math.radians(degrees), original_origin, rotated_origin)
        
        xmin = polygon.xmin()
        xmax = polygon.xmax()
        ymin = polygon.ymin()
        ymax = polygon.ymax()

        box = PointSet2D([ymin, ymax, ymax, ymin], [xmin, xmin, xmax, xmax])
        rotated_boxes.append(box)
        
        rotated_box_back = box.rotate_around_point(math.radians(-degrees, rotated_origin, original_origin))
        boxes.append(rotated_box_back)
        
    return boxes, rotated_boxes, rotated_image

