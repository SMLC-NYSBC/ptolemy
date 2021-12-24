import os
import sys
import json

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path)

import numpy as np
import torch

from ptolemy.images import load_mrc, Exposure
import ptolemy.algorithms as algorithms
import ptolemy.models as models
modelpath = path + 'weights/211215_lowmag_64x5_defaultadam_tightw_e2.torchmodel'

def main():

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('path', help='path to montage image')
    parser.add_argument('-v', action='store_true')
    parser.add_argument('-f', '--format', dest='format_', default='txt', help='format to write region coordinates')
    parser.add_argument('-o', '--output', help='output file path')

    args = parser.parse_args()
    verbose = args.v
    path = args.path
    format_ = args.format_
    output_path = args.output

    # open the montage
    image = load_mrc(path)
#     if len(image.shape) > 2:
#         print('WARNING: ' + path + ' is an image stack. Only processing the first image.', file=sys.stderr)
#         image = image[0]
    ex = Exposure(image)
    
    segmenter = algorithms.PMM_Segmenter()
    ex.make_mask(segmenter)
    
    processor = algorithms.LowMag_Process_Mask()
    ex.process_mask(processor)
    
    cropper = algorithms.LowMag_Process_Crops()
    ex.get_crops(cropper)
    model = models.LowMag_64x5_2ep()
    model.load_state_dict(torch.load(modelpath))
    wrapper = models.Wrapper(model)
    ex.score_crops(wrapper)
    
    vertices = [box.as_matrix_y().tolist() for box in ex.crops.boxes]
    areas = [box.area() for box in ex.crops.boxes]
    centers = np.round(ex.crops.center_coords.as_matrix_y()).astype(int).tolist()
    intensities = ex.mean_intensities
    scores = ex.crops.scores
    
    if format_ == 'json':
        order = np.argsort(scores)[::-1]
        js = []
        for i in order:
            d = {}
            d['vertices'] = vertices[i]
            d['center'] = centers[i]
            d['area'] = float(areas[i])
            d['brightness'] = float(intensities[i])
            d['score'] = float(scores[i])
            
            js.append(d)
        
        content = json.dumps(js)

    elif format_ == 'txt': # write coordinates simply as tab delimited file
        # get the regions centers
        points = centers
        # points are (y-axis, x-axis)
        # flip to (x-axis, y-axis)
        # points = np.stack([points[:,1], points[:,0]], axis=1)

        content = ['\t'.join(['x_coord', 'y_coord'])]
        for point in points:
            content.append('\t'.join([str(point[0]), str(point[1])]))
        content = '\n'.join(content)

    f = sys.stdout
    if args.output is not None:
        f = open(args.output, 'w')
    print(content, file=f)


if __name__ == '__main__':
    main()

