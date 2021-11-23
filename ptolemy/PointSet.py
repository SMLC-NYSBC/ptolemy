import numpy as np
import math

class PointSet2D():
    def __init__(self, y, x):
        if type(x) is list:
            x = np.array(x)
        if type(y) is list:
            y = np.array(y)
        self.x = x            
        self.y = y

    def area(self):
        # compute area treating pointset as polygon
        area = 0.5*np.abs(np.dot(self.x, np.roll(self.y, 1)) - np.dot(self.y, np.roll(self.x, 1)))
        return area

    def center_of_mass(self):
        # compute center of mass of point set
        v = (self.x*np.roll(self.y, 1) - self.y*np.roll(self.x, 1))
        cx = np.sum((self.x + np.roll(self.x, 1))*v)
        cy = np.sum((self.y + np.roll(self.y, 1))*v)
        a = np.sum(v)

        return np.array([cy/3/a, cx/3/a])

    def rotate_around_point(self, radians, init_origin, new_origin, inplace=False):
        if type(init_origin) is int:
            init_origin = [init_origin, init_origin]
        if type(new_origin) is int:
            new_origin = [new_origin, new_origin]

        adj_y = self.y - init_origin[0]
        adj_x = self.x - init_origin[1]

        cos_rad = math.cos(radians)
        sin_rad = math.sin(radians)

        y = init_origin[1] + -sin_rad * adj_x + cos_rad * adj_y
        x = init_origin[0] + cos_rad * adj_x + sin_rad * adj_y

        y = y + new_origin[1] - init_origin[1]
        x = x + new_origin[0] - init_origin[0]

        if inplace:
            self.x = x
            self.y = y
        else:
            return PointSet2D(y, x)

    def xmin(self):
        return min(self.x)
    
    def xmax(self):
        return max(self.x)

    def ymin(self):
        return min(self.y)

    def ymax(self):
        return max(self.y)

    def get_bounding_box(self):
        y = [self.ymin(), self.ymax(), self.ymax(), self.ymin()]
        x = [self.xmin(), self.xmin(), self.xmax(), self.xmax()]
        
        return PointSet2D(y, x)