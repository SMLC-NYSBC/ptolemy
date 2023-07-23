import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from ptolemy.PointSet import PointSet2D


# TODO add functionalty for visualizing radii, 
def viz_lm_image(image, boxes=None, scores=None, operator_selections=None, centers=None):
    # Assumes boxes is a list of pointsets, should probably set that if it's not, or rationalize this generally
    # also assumes operator_selections is a pointset

    _, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image, cmap='Greys_r')

    if scores is not None and boxes is not None:
        assert len(scores) == len(boxes)

    if boxes is not None:
        patches = []
        for box in boxes:
            patches.append(matplotlib.patches.Polygon(box, facecolor='None'))
        
        collection = matplotlib.collections.PatchCollection(patches)

        if scores is not None:
            cmap = plt.get_cmap('RdYlBu')
            colors = np.array(scores)
            collection.set_color(cmap(colors / np.max(colors)))
            collection.set_facecolor('none')
            collection.set_linewidth(2)

        else:
            collection.set_color('r')
            collection.set_facecolor('none')
            collection.set_linewidth(2)

        ax.add_collection(collection)

    plt.axis('off')
    
    if operator_selections is not None and ((type(operator_selections) == list) or (type(operator_selections) == np.ndarray)):
        # assumes as_matrix_y
        x = [selection[1] for selection in operator_selections]
        y = [selection[0] for selection in operator_selections]
        ax.scatter(x, y)
    elif operator_selections and operator_selections is PointSet2D:
        ax.scatter(operator_selections.x, operator_selections.y)

    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1])


def viz_mm_image(image, centers=None, radii=None, scores=None, operator_selections=None, print_scores=False):
    # Assumes boxes is a list of pointsets, should probably set that if it's not, or rationalize this generally
    # also assumes operator_selections is a pointset
    if radii is not None:
        assert centers is not None

    _, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image, cmap='Greys_r')

    if centers is not None and (((type(centers) == list) or (type(centers) == np.ndarray) or (type(centers) == PointSet2D))):
        # assumes as_matrix_y
        if type(centers) == list or type(centers) == np.ndarray:
            center_x = [center[0] for center in centers]
            center_y = [center[1] for center in centers]
        else:
            center_x = operator_selections.x
            center_y = operator_selections.y
        
        if scores is not None and print_scores:
            for cx, cy, score in zip(center_x, center_y, scores):
                plt.text(cx, cy, round(score, 2), bbox=dict(boxstyle='round', facecolor='white'))
        elif scores is not None:
            ax.scatter(center_x, center_y, cmap='RdYlGn', c=scores)
        else:
            ax.scatter(center_x, center_y)
    
    if radii is not None:
        patches = []
        if type(radii) == list:
            for x, y, radius in zip(center_x, center_y, radii):
                patches.append(matplotlib.patches.Circle((x, y), radius))
        else:
            for x, y in zip(center_x, center_y):
                patches.append(matplotlib.patches.Circle((x, y), radii))

        collection = matplotlib.collections.PatchCollection(patches)

        if scores is not None:
            cmap = plt.get_cmap('RdYlGn')
            if type(scores) is not np.ndarray:
                scores = np.array(scores)
            collection.set_color(cmap(scores))
            collection.set_facecolor('none')
            collection.set_linewidth(2)

        else:
            collection.set_color('r')
            collection.set_facecolor('none')
            collection.set_linewidth(2)
        
        ax.add_collection(collection)

    plt.axis('off')

    if operator_selections is not None and ((type(operator_selections) == list) or (type(operator_selections) == np.ndarray)):
        # assumes as_matrix_y
        x = [selection[1] for selection in operator_selections]
        y = [selection[0] for selection in operator_selections]
        plt.scatter(x, y)
    elif operator_selections and operator_selections is PointSet2D:
        plt.scatter(operator_selections.x, operator_selections.y)