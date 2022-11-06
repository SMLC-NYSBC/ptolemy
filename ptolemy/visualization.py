import matplotlib.pyplot as plt
import matplotlib
import numpy as np


# TODO add functionalty for visualizing radii, 
def viz_image(self, image, boxes=None, scores=None, operator_selections=None):
    # Assumes boxes is a list of pointsets, should probably set that if it's not, or rationalize this generally
    # also assumes operator_selections is a pointset

    _, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image, cmap='Greys_r')

    if scores and boxes:
        assert len(scores) == len(boxes)

    if boxes:
        patches = []
        for box in boxes:
            patches.append(matplotlib.patches.Polygon(box.as_matrix_y(), facecolor='None'))
            collection = matplotlib.collections.PatchCollection(patches)


        if scores:
            cmap = plt.get_cmap('RdYlBu')
            colors = np.array(scores)
            collection.set_color(cmap(colors / np.max(colors)))
            collection.set_facecolor('none')
            collection.set_linewidth(2)

        ax.add_collection(collection)
        collection.set_color('r')
        collection.set_facecolor('none')
        collection.set_linewidth(2)

    plt.axis('off')
    
    if operator_selections:
        plt.scatter(operator_selections.x, operator_selections.y)


