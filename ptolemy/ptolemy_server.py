from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from typing import List
import io

import argparse

import numpy as np
import pandas as pd
from ptolemy.Ptolemy import Ptolemy
from ptolemy.Ptolemy_AL import Ptolemy_AL
from ptolemy.mrc import load_mrc
from ptolemy.utils import prep_state_for_csv

# parser = argparse.ArgumentParser()
# parser.add_argument('-c', '--config', default='default', help='path to config file')
# parser.add_argument('-l', '--lm-state', dest='lm_state', help='path to pickled dataframe with lowmag state')
# parser.add_argument('-m', '--mm-state', dest='mm_state', help='path to pickled dataframe with med state')

# args = parser.parse_args()
# print(parser.mm_state)
# print(parser.lm_state)

app = FastAPI()
base_model = Ptolemy() # Either in setup you must set this manually or I gotta figure out how to pass arguments into a fastapi server 
al_model = Ptolemy_AL() # same goes here

##### Request Data Types

class image(BaseModel):
    """
    For use when we want to do stateless processing
    """
    image: list

class lm_image(BaseModel):
    grid_id: int
    tile_id: int = None
    image: list

class mm_image(BaseModel):
    grid_id: int
    square_id: int
    mm_img_id: int = None
    image: list

class path(BaseModel):
    path: str

class pathlist(BaseModel):
    paths: list

class single_float(BaseModel):
    value: float

class single_int(BaseModel):
    value: int

# should we allow this to be batched? Probably. 
# just gotta switch this over to a list
class visit_hole_record(BaseModel):
    hole_ids: list # list somehow
    ctfs: list 
    ice_thicknesses: list

class visit_square_record(BaseModel):
    square_id: int

class list_of_ints(BaseModel):
    ints: list


class init_new_session(BaseModel):
    new_state_path: str = None
    historical_state_paths: list = []
    save_state_path: str = None


@app.get('/')
def main():
    return {'message': 'Welcome'}

# @app.get('/{name}')
# def hello_name(name: str):
#     return {'message': f'Welcome, {name}'}

@app.get('/predict')
def predict():
    # return {'image': data.image}
    # basic_df = pd.DataFrame({'x': [1, 2, 3], 'n': [4, 5, 6]})
    # image = pd.read_csv(io.StringIO(data.path), index_col='x')
    # return {'image': image.to_csv()}
    return [0, 1, 2, 3]

@app.post('/set_config')
def set_config(data: path):
    """
    Sets the model and al configs based on path
    path should probably be an absolute path
    """
    base_model.load_config_and_models(data.path)
    al_model.load_config(data.path)


@app.post('/load_historical_state')
def append_historical_state(data: path):
    """
    Appends state at path to historical state. Currently assumes path points to a pickled dataframe
    """
    al_model.append_historical_state(data.path)


@app.post('/append_multi_historical_states')
def append_multi_historical_states(data: pathlist):
    for path in data.paths:
        al_model.append_historical_state(path)


@app.post('/overwrite_current_state')
def overwrite_current_state(data: path):
    """
    Overwrites current state with state at path
    """
    al_model.overwrite_current_state(data.path)


@app.get('/clear_current_state')
def clear_current_state():
    """
    Clears the current state
    """
    al_model.clear_current_state()


@app.get('/clear_historical_state')
def clear_historical_state():
    al_model.clear_historical_state()


@app.post('/initialize_new_session')
def initialize_new_session(data: init_new_session):
    """
    noice_hole_intensity null value is -1 (since there should never be a negative intensity). 
    """
    al_model.initialize_new_session(data.new_state_path, data.historical_state_paths, data.save_state_path)
    base_model.update_noice_hole_intensity(-1)

@app.get('/initialize_new_session')
def initialize_new_session():
    """
    noice_hole_intensity null value is -1 (since there should never be a negative intensity). 
    """
    al_model.initialize_new_session()
    base_model.update_noice_hole_intensity(-1)


@app.post('/append_current_state')
def append_current_state(data: path):
    """
    Appends state at path to current state
    This should be done with caution. Ideally, current state should always be in one dataframe, and should always be saved and loaded as a complete whole, so only overwrite_current_state should be needed. Appending here means you have squares for the current session (accessible on this grid/cassette) that were saved across multiple states. 
    """
    al_model.append_current_state(data.path)


@app.post('/save_state')
def save_state(data: path):
    """
    Saves the current AL state
    """
    al_model.save_state(data.path)


@app.post('/process_stateless_lm')
def process_stateless_lm(data: image):
    """
    Process a low-mag image and return everything that the old CLI would have returned plus features and prior scores
    """
    image = np.array(data.image)

    # check shapes

    raw_crops, centers, vertices, areas, mean_intensities, features, prior_scores = base_model.process_lm_image(image)

    order = np.argsort(prior_scores)[::-1]
    js = []
    for i in order:
        d = {}
        d['vertices'] = vertices[i]
        d['center'] = centers[i]
        d['area'] = float(areas[i])
        d['brightness'] = float(mean_intensities[i])
        d['score'] = float(prior_scores[i])
        d['features'] = features[i].tolist()

        js.append(d)
    
    return js


@app.post('/process_stateless_mm')
def process_stateless_mm(data: image):
    """
    Process a medium-mag image and return everything that the old CLI would have returned plus features and prior scores
    """
    image = np.array(data.image)

    # check shapes

    crops, centers, boxes, radii, features, prior_scores = base_model.process_mm_image(image)

    order = np.argsort(prior_scores)[::-1]
    js = []
    for i in order:
        d = {}
        d['vertices'] = boxes[i].tolist()
        d['center'] = centers[i]
        d['score'] = float(prior_scores[i])
        d['radius'] = float(radii[i])
        d['features'] = features[i].tolist()

        # probably have to verify types here

        js.append(d)
    
    return js


@app.post('/select_next_square') # need a better name for this
def select_next_square(data: single_int):
    """
    Asks the server to return the dataframe with the information for picking the next square

    single_int should contain int index of grid to use, or -1 if use all grids

    """
    unvisited_squares = al_model.run_lm_gp(data.value)
    """
    From here we can either return the entire dataframe or just the square id
    Let's default to returning the whole dataframe

    On the other end, to correctly unpack this dataframe, call
    pd.read_csv(io.StringIo(request.json()), index_col='square_id')
    """
    return prep_state_for_csv(unvisited_squares).to_csv()


@app.post('/push_and_evaluate_mm')
def push_and_evaluate_mm(data: mm_image):
    # check to see if noice_hole_intensity has been set manually. If not, send a warning that we are using some default value.

    image = np.array(data.image)

    # check shapes here

    crops, centers, boxes, radii, features, prior_scores = base_model.process_mm_image(image)

    holes_to_run = []
    for center, feature, prior_score, radius in zip(centers, features, prior_score, radii):
        hole_id = al_model.add_hole_to_state(data.square_id, data.grid_id, data.mm_img_id, center, feature, prior_score, radius)
        holes_to_run.append(hole_id)

    hole_results = al_model.run_mm_gp(hole_ids=holes_to_run)
    al_model.set_active_holes(holes_to_run)

    return prep_state_for_csv(hole_results).to_csv()


@app.get('/rerun_mm_on_active_holes')
def rerun_mm_on_active_holes():
    hole_results = al_model.run_mm_gp(active=True)
    return prep_state_for_csv(hole_results).to_csv()

 
@app.post('/rerun_mm_on_arbitrary_holes')
def rerun_mm_on_arbitrary_holes(data: list_of_ints):
    hole_results = al_model.run_mm_gp(hole_ids = data.ints)
    return prep_state_for_csv(hole_results).to_csv()


@app.post('/push_lm')
def push_lm(data: lm_image): 
    image = np.array(data.image)

    # check shapes here

    raw_crops, centers, vertices, areas, mean_intensities, features, prior_scores = base_model.process_lm_image(image)

    for center, feature, prior_score, vertex_set, brightness, area in zip(centers, features, prior_scores, vertices, mean_intensities, areas):
        al_model.add_square_to_state(data.grid_id, data.tile_id, center, feature, prior_score, vertex_set, brightness, area)


@app.post('/push_mm')
def push_mm(data: mm_image):
    # check to see if noice_hole_intensity has been set manually. If not, send a warning that we are using some default value.

    image = np.array(data.image)

    # check shapes here

    crops, centers, boxes, radii, features, prior_scores = base_model.process_mm_image(image)

    for center, feature, prior_score, radius in zip(centers, features, prior_scores, radii):
        al_model.add_hole_to_state(data.square_id, data.grid_id, data.mm_img_id, center, feature, prior_score, radius)


@app.post('/visit_holes')
def visit_holes(data: visit_hole_record):
    for hole_id, ctf, ice_thickness in zip(data.hole_ids, data.ctfs, data.ice_thicknesses):
        al_model.visit_hole(hole_id, ctf, ice_thickness)
    
    al_model.active_holes = al_model.active_holes - set(data.hole_ids)


@app.get('/current_lm_state')
def current_lm_state():
    return prep_state_for_csv(al_model.current_lm_state).to_csv()


@app.get('/current_mm_state')
def current_mm_state():
    return prep_state_for_csv(al_model.current_mm_state).to_csv()


@app.post('/visit_square')
def visit_square(data: visit_square_record):
    al_model.visit_square(data.square_id)


@app.post('/set_noice_hole_intensity')
def set_noice_hole_intensity(data: single_float):
    base_model.update_noice_hole_intensity(data.value)




# But assuming this is true, (and I think it is) I think ideally we expose an API on the same machine that individually exposes endpoints for
# - Initialize session
#     - ingest user inputs, clear current square and hole dataset / replace them with them optionally passed
# - Push grid images
#     - clear current candidate squares, ingest a set of tiled grid images, run the low-mag segmenter (and prior model classifier?), and get a bunch of square crops and locations but don’t return them
# - get full grid eval
#     - return all square locations and corresponding probabilities and GP estimates
# - get next square
#     - run the GP/prior and return the coordinates of the next square that the model thinks we should explore
# - pick holes in mm image
#     - run the segmentation, GP and hole contaminant models, and output the coordinates of the hole centers that the model should collect
# - evaluate medium-mag image and return all information
#     - run the segmentation, GP and hole contaminant models, and then output all hole centers, GP predictions and hole contaminant probabilities.
# - thumb up thumbs down this hole
#     - given hole coordinates, run the GP and hole contaminant models and return probabilities or just a collect/don’t collect signal
# - push this hole / these holes
#     - give Ptolemy a set of hole coordinates and respective ctf/ice for Ptolemy to include in it’s GPs
# - push this square / these squares
#     - give Ptolemy a set of square coordinates and respective ctf counts.





"""
Let's imagine a data collection workflow first, then think about a screening workflow.


First, you push a bunch of lm's, then ask for the next square

You then give a medium-mag image of the square, and you ask the model to tell you which holes to skip. Let's call that "push_and_eval_mm".

You then visit the holes, so you wanna be able to tell the model which holes you ended up visiting.

The holes and squares are marked as visited, and you then you ask for the next square, and repeat

Another function we should throw in is the model's estimate of the number of holes left of various ctfs in the grid. 

So this requires

-push_and_eval_mm



"""