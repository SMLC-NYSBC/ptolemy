import requests
import time
from multiprocessing import Process
from uvicorn import Config, Server
import uvicorn

import os
import pandas as pd
import pickle
from ptolemy.mrc import load_mrc
from ptolemy.visualization import viz_mm_image, viz_lm_image
import numpy as np
import matplotlib.pyplot as plt
from ptolemy.Requester import Ptolemy_Requester
import matplotlib.path as mplPath
from MicroscopeSimulator import Microscope_Simulator


# This function will run the Uvicorn server
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, required=True)
args = parser.parse_args()

def run_uvicorn():
    uvicorn.run("ptolemy.ptolemy_server:app", host="127.0.0.1", port=args.port)

    # global server
    # config = Config("ptolemy.ptolemy_server:app", host="127.0.0.1", port=args.port, log_level="info")
    # server = Server(config=config)
    # server.run()

# Start the server in a separate thread
# thread = threading.Thread(target=run_uvicorn)
# thread.start()
server = Process(target=run_uvicorn)
server.start()

# Give the server some time to start
time.sleep(20)

sim = Microscope_Simulator()

# Send a request to the server
requester = Ptolemy_Requester(f'http://127.0.0.1:{args.port}/')
requester.set_config("/h2/pkim/ptolemy/ptolemy/test_lm_config_gpu.json")
requester.initialize_new_session('m23feb16a/first_square_visited.state')
requester.set_noice_hole_intensity(13.882552)

# Give the server some time to process
time.sleep(3)

hole_ctfs = []
square_hole_ctfs = []
current_mm_state = requester.get_current_mm_state()
hole_ctfs.extend(list(current_mm_state[current_mm_state.visited].ctf))
square_hole_ctfs.append(list(current_mm_state[current_mm_state.visited].ctf))

start = time.time()

progress = 1

while progress < 460:
    df = requester.select_next_square(1)
    df['posterior'] = df.prior_score * df.GP_probs
    df = df.sort_values(by='posterior', ascending=False)
    for squareid, row in df.iterrows():
        square_images = sim.get_square_images(coordinates=row.vertices, grid_id=row.grid_id, tile_id=row.tile_id)
        if 'failed' not in square_images.keys():
            break
    if 'failed' in square_images.keys():
        break
        
    print('visiting square {}/{}, time {}'.format(progress, 426, round(time.time()-start)))
    
    this_square_ctfs = []
    for sq_img_id, sq_img in square_images.items():
        res = requester.push_and_evaluate_mm(sq_img, 1, squareid, sq_img_id)
        hole_ids, ctfs, ices = [], [], []
        for hole_id, row in res.iterrows():
            if row.prior_score < 0.75:
                continue
            ice_ctf = sim.get_ice_ctf(row.center_x, row.center_y, row.radius, row.mm_img_id)
            if 'failed' not in ice_ctf.keys():
                hole_ids.append(hole_id)
                ctfs.append(ice_ctf['ctf'])
                ices.append(ice_ctf['ice'])
        requester.visit_holes(hole_ids, ctfs, ices)
        hole_ctfs.extend(ctfs)
        this_square_ctfs.extend(ctfs)
        print('visited {} holes'.format(len(hole_ids)))
    
    square_hole_ctfs.append(this_square_ctfs)
    progress += 1
        
pickle.dump([hole_ctfs, square_hole_ctfs], open('m23feb16a_results/230610_testhparam_cutoff3.5_squareposterior_holeprior_gt_75.pkl', 'wb'))

# Stop the server
# This is done by setting the running state of the loop in the server to False
# server.lifespan._state.set(False)
server.terminate()
server.join()

# Give the server some time to stop
# time.sleep(30)

print('ending')
# thread.end()
