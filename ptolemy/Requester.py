import os
import sys
import json
import numpy as np
import requests
import io
import pandas as pd

from ptolemy.utils import clean_vertex_columns

"""
A class with convenience functions for making requests and unpacking responses to the Ptolemy Server
"""

class Ptolemy_Requester():
    def __init__(self, baseurl):
        """
        Initialize with the url at which the Ptolemy server is listening
        """
        self.baseurl = baseurl


    def post_request(self, request_name, payload):
        return requests.post(self.baseurl + request_name, json.dumps(payload))


    def get_request(self, request_name):
        return requests.get(self.baseurl + request_name)

    
    def read_csv(self, response, index_col):
        return pd.read_csv(io.StringIO(response.json()), index_col=index_col)


    def set_noice_hole_intensity(self, noice_hole_intensity: float):
        payload = {
            'value': noice_hole_intensity
        }
        self.post_request('set_noice_hole_intensity', payload)

    
    def set_config(self, path):
        self.pass_path('set_config', path)


    def push_lm(self, image, grid_id : int, tile_id : int = None):
        payload = {
            'image': image.tolist(),
            'grid_id': grid_id,
            'tile_id': tile_id
        }
        self.post_request('push_lm', payload)

    
    def select_next_square(self, grid_id: int):
        payload = {
            'value': grid_id
        }
        response = self.post_request('select_next_square', payload)
        df = self.read_csv(response, 'square_id')
        return clean_vertex_columns(df)

    
    def push_and_evaluate_mm(self, image, grid_id: int, square_id: int, mm_img_id: int = None):
        payload = {
            'image': image.tolist(),
            'grid_id': grid_id,
            'square_id': square_id,
            'mm_img_id': mm_img_id
        }
        response = self.post_request('push_and_evaluate_mm', payload)
        return self.read_csv(response, 'hole_id')

    
    def visit_holes(self, hole_id, ctf, ice_thickness):
        """
        hole_id: int or list of ints
        ctf: float or list of floats
        ice_thickness: float or list of floats
        """

        if hole_id is list:
            payload = {
                'hole_ids': hole_id,
                'ctfs': ctf,
                'ice_thicknesses': ice_thickness
            }
        elif hole_id is int:
            payload = {
                'hole_ids': [hole_id],
                'ctfs': [ctf],
                'ice_thicknesses': [ice_thickness]
            }

        self.post_request('visit_holes', payload)


    def rerun_mm_on_active_holes(self):
        response = self.get_request('rerun_mm_on_active_holes')
        return self.read_csv(response, 'hole_id')


    def process_stateless_lm(self, image):
        payload = {
            'image': image.tolist()
        }
        response = self.post_request('process_stateless_lm', payload)
        return response.json()

    
    def process_stateless_mm(self, image):
        payload = {
            'image': image.tolist()
        }
        response = self.post_request('process_stateless_mm', payload)
        return response.json()


    def pass_path(self, request_name, path):
        payload = {
            'path': path
        }
        self.post_request(request_name, payload)


    def set_config(self, path):
        self.pass_path('set_config', path)


    def load_historical_state(self, path):
        self.pass_path('load_historical_state', path)

    
    def overwrite_current_state(self, path):
        self.pass_path('overwrite_current_state', path)


    def clear_current_state(self):
        self.get_request('clear_current_state')

    
    def clear_historical_state(self):
        self.get_request('clear_historical_state')

    
    def initialize_new_session(self, new_state_path=None, historical_state_paths=[], save_state_path=None):
        """
        new_state_path: optional path to the state that should be loaded as the current_state
        historical_state_paths: optional list of paths that should be loaded as historical states
        save_state_path: optional path to which the current state should be saved
        """
        payload = {
            'new_state_path': new_state_path,
            'historical_state_paths': historical_state_paths,
            'save_state_path': save_state_path
        }
        self.post_request('initialize_new_session', payload)

    
    def save_state(self, path):
        self.pass_path('save_state', path)

    
    def get_current_lm_state(self):
        lm_state = self.get_request('current_lm_state')
        lm_state = self.read_csv(lm_state, 'square_id')
        return clean_vertex_columns(lm_state)

    
    def get_current_mm_state(self):
        mm_state = self.get_request('current_mm_state')
        return self.read_csv(mm_state, 'hole_id')

    
    
    


