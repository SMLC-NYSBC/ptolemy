# Contain simple geometry functions
from PointSet import PointSet2D
import numpy as np
from scipy.spatial import ConvexHull


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



