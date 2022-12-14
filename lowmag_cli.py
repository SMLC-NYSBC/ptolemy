import os
import sys
import json

path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, path)

import numpy as np

from ptolemy.Ptolemy import Ptolemy
from ptolemy.mrc import load_mrc


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('path', help='path to montage image')
    parser.add_argument('-v', action='store_true', help='verbose')
    parser.add_argument('-f', '--format', dest='format_', default='json', help='format to write region coordinates')
    parser.add_argument('-o', '--output', help='output file path')
    parser.add_argument('-c', '--config', default='default', help='path to config file')

    args = parser.parse_args()
    verbose = args.v
    path = args.path
    format_ = args.format_
    output_path = args.output
    config = args.config

    pipeline = Ptolemy(config)

    # open the montage
    image = load_mrc(path)
    
    outputs = pipeline.process_lm_image(image)

    _, centers, vertices, areas, intensities, _, scores = outputs
    
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
        f = open(output_path, 'w')
    print(content, file=f)


if __name__ == '__main__':
    main()

