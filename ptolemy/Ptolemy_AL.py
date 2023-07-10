import json
import os

import numpy as np
import pandas as pd
import torch
import gpytorch
import time

from ptolemy.utils import prep_state_for_csv


class SingleTaskGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SingleTaskGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=4))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# class ExactGPModel_Priors(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood, meanprior, outputscaleprior):
#         super(ExactGPModel_Priors, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMean(meanprior)
#         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=4), outputscaleprior)

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class MultitaskGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_tasks):
        super(MultitaskGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), num_tasks=n_tasks)
        self.covar_module = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.RBFKernel(), num_tasks=n_tasks, rank=n_tasks)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


def upper_confidence_bound(dist, n_samples=10000):
    sample = dist.sample_n(n_samples)
    sample_max = sample.argmax(axis=1)
    probs = np.zeros(len(dist.loc))
    for sample in sample_max:
        probs[sample] += 1
    return probs / n_samples

def upper_confidence_bound_v2(dist, n_samples=10000):
    sample = dist.sample_n(n_samples)
    sample_mean = sample.mean(axis=0)
    sample_std = sample.std(axis=0)
    upperbound = sample_mean + (2 * sample_std)
    probs = upperbound / max(upperbound)
    return probs


class Ptolemy_AL:
    """
    Class that manages running the active learning algorithms for Ptolemy to update scores on-the-fly. 

    Holds state for low-mag and medium-mag cases

    There are two types of states to consider. "Historical" state and "Current" state. 
    
        - Historical states are from previous grids whose data is relevant for the current grid. Their grid_ids, hole_ids, and square_ids are meaningless, but the data is used in addition to the current state to fit the GPs.
        - Current state is the current grid or cassette that is being operated on. These grid_ids, hole_ids, and square_ids are relevant and potentially accessible for collection. Current state is updated during collection, and can be saved to be used later as a historical state. You should load current state when, for example, you want to resume collection on a grid. 

        Currently, there is no priority on the GPs to weight "current" state data higher than "historical" state data, so be careful about what data you put into "historical" state. 


    Initially we are going to use fixed GP hyperparameters.
    At some point should probably do some experiments that show this is better
    """

    def __init__(self, config='default', historical_state_paths=[], current_state_path=None):
        self.historical_lm_state = []
        self.historical_mm_state = []

        for path in historical_state_paths:
            self.append_historical_state(path)

        if current_state_path is None:
            self.current_lm_state = pd.DataFrame(columns= ['square_id', 'tile_id', 'grid_id', 'center_x', 'center_y', 'features', 'prior_score', 'visited', 'vert_1_x', 'vert_1_y', 'vert_2_x', 'vert_2_y', 'vert_3_x', 'vert_3_y', 'vert_4_x', 'vert_4_y', 'brightness', 'area'])
            self.current_lm_state = self.current_lm_state.set_index('square_id')

            self.current_mm_state = pd.DataFrame(columns= ['hole_id', 'mm_img_id', 'square_id', 'grid_id', 'center_x', 'center_y', 'features', 'radius', 'prior_score', 'visited', 'ctf', 'ice_thickness'])
            self.current_mm_state = self.current_mm_state.set_index('hole_id')
        else:
            self.current_lm_state, self.current_mm_state = self.load_state(current_state_path)

        if config == 'default':
            config_path = os.path.dirname(os.path.realpath(__file__)) + '/default_config_cpu.json'
        else:
            config_path = config
        
        self.load_config(config_path)
        self.current_active_holes = set()
        self.device = self.settings['device']
        
        self.modeling_cap = None

    
    def load_config(self, config_path):
        self.settings = json.load(open(config_path, 'r'))


    def load_state(self, path):
        lm_state = pd.read_csv(path + '/lmstate.csv', index_col='square_id')
        mm_state = pd.read_csv(path + '/mmstate.csv', index_col='hole_id')

        lm_feats = np.load(path + '/lm_feats.npy')
        lm_state['features'] = list(lm_feats)
        mm_feats = np.load(path + '/mm_feats.npy')
        mm_state['features'] = list(mm_feats)

        return lm_state, mm_state

    
    def append_historical_state(self, path):
        lm_state, mm_state = self.load_state(path)
        self.historical_lm_state.append(lm_state)
        self.historical_mm_state.append(mm_state)


    def save_state(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)

        prep_state_for_csv(self.current_lm_state).to_csv(path + '/lmstate.csv')
        prep_state_for_csv(self.current_mm_state).to_csv(path + '/mmstate.csv')
        np.save(path + '/lm_feats.npy', np.stack(self.current_lm_state.features.values))
        np.save(path + '/mm_feats.npy', np.stack(self.current_mm_state.features.values))



    def overwrite_current_state(self, path):
        self.current_lm_state, self.current_mm_state = self.load_state(path)

    
    def clear_current_state(self):
        self.current_lm_state = self.current_lm_state[0:0]
        self.current_mm_state = self.current_mm_state[0:0]

    
    def clear_historical_state(self):
        self.historical_lm_state = []
        self.historical_mm_state = []

    
    def set_device(self, device):
        self.device = device


    def initialize_new_session(self, new_state_path=None, historical_state_paths=[], save_state_path=None):
        """
        clears the current state and historical state, optionally saves the current state and loads new current and historical states.         
        """
        if save_state_path: self.save_state(save_state_path)

        self.clear_current_state()
        if new_state_path: self.current_lm_state, self.current_mm_state = self.load_state(new_state_path)
        
        self.clear_historical_state()
        if len(historical_state_paths) > 0: [self.append_historical_state(path) for path in historical_state_paths]


    def append_current_state(self, path):
        """
        This should be used with caution. Ideally, current state should always be in one dataframe, and should always be saved and loaded as a complete whole. Appending here means you have squares for the current session (accessible on this grid/cassette) that were saved across multiple states. 
        """    

        # TODO raise a warning
        lm_state, mm_state = self.load_state(path)
        self.current_lm_state = pd.concat((self.current_lm_state, lm_state))
        self.current_mm_state = pd.concat((self.current_mm_state, mm_state))


    def compute_lengthscale(self):
        # Low mag lengthscale is computed just based on the square features alone. 
        # No visiting required.
        features = np.stack(self.current_lm_state.features.values)
        if len(self.historical_lm_state) > 0:
            features_cat = []
            for lm_state in self.historical_lm_state:
                features_cat.append(np.stack(lm_state.features.values))

            features_cat.append(features)
            features = np.cat(features_cat)
        
        lengthscale = np.quantile(features, q=0.25, axis=0)
        return torch.tensor(lengthscale).unsqueeze(0)


    def _set_lm_parameters(self, model):
        # model.mean_module.constant = torch.nn.Parameter(torch.tensor([float(self.settings['lm_gp_mean_constant'])])) # default should be 20 for now
        model.mean_module.constant = float(self.settings['lm_gp_mean_constant'])
        all_square_features = torch.tensor(np.stack(self.current_lm_state[self.current_lm_state.prior_score > 0.2]['features'].values).astype('float'))
        all_square_features = (all_square_features - all_square_features.mean(dim=0)) / all_square_features.var(dim=0)
        model.covar_module.base_kernel.raw_lengthscale = torch.nn.Parameter(torch.quantile(all_square_features, q=0.25, dim=0).unsqueeze(0).float())
        model.likelihood.noise_covar.raw_noise = torch.nn.Parameter(torch.tensor([float(self.settings['lm_gp_noise_constant'])])) # default should be 15
        model.covar_module.outputscale = float(self.settings['lm_gp_outputscale']) # default should be 500
        
        return model
    
    # def baseline_model_state_dict(self, train_x, train_y):
    #     likelihood = gpytorch.likelihoods.GaussianLikelihood()
    #     model = SingleTaskGP(train_x, train_y, likelihood)

    #     model.mean_module.constant = torch.nn.Parameter(torch.tensor([float(self.settings['mean_constant'])])) # default should be 20 for now
    #     model.covar_module.base_kernel.raw_lengthscale = torch.nn.Parameter(torch.quantile(train_x, q=0.25, dim=0).unsqueeze(0))
    #     model.likelihood.noise_covar.raw_noise = torch.nn.Parameter(torch.tensor([float(self.settings['noise_constant'])])) # default should be 15
    #     model.covar_module.outputscale = float(self.settings['outputscale']) # default should be 500

    #     save_dict = model.state_dict()
        
    #     return save_dict

    def run_lm_gp(self, grid_id=-1):
        # Get visited squares
        # Create ctf-based training set
        # initially use ctf < 5 as cutoff - in the future, modify this to allow for multitask lm model
        # Fit GP with set hyperparameters
        # Predict unvisited squares, compute UCB probabilities
        # return dataframe of unvisited squares with gp probabilities
        assert len(self.current_lm_state) > 0, "must have pushed lm images"
        
        if len(self.current_mm_state) == 0:
            unvisited_squares = self.current_lm_state[~self.current_lm_state['visited']]
            if grid_id != -1:
                unvisited_squares = unvisited_squares[unvisited_squares['grid_id'] == grid_id]
            
            unvisited_squares['GP_probs'] = 1 / len(unvisited_squares)
            return unvisited_squares
            
        

        train_x, train_y = [], []

        visited_holes = self.current_mm_state[self.current_mm_state['visited']].dropna(subset=['features', 'ctf', 'ice_thickness'])
        visited_squares = self.current_lm_state[self.current_lm_state['visited']].dropna(subset=['features'])
                
        for square_id, row in visited_squares.iterrows():
            train_x.append(row.features)
            holes = visited_holes[visited_holes.square_id == square_id]
            counts = (holes.ctf < self.settings["lm_ctf_good_hole_cutoff"]).sum()
            train_y.append(counts)

        for lm_state, mm_state in zip(self.historical_lm_state, self.historical_mm_state):
            for square_id, row in lm_state[lm_state['visited']].dropna(subset=['features']):
                train_x.append(row.features)
                visited_holes = mm_state[mm_state['visited']].dropna(subset=['features', 'ctf', 'ice_thickness'])
                holes = visited_holes[visited_holes.square_id == square_id]
                counts = (holes.ctf < self.settings["lm_ctf_good_hole_cutoff"]).sum()
                train_y.append(counts)
                
        unvisited_squares = self.current_lm_state[~self.current_lm_state['visited']]
        if grid_id != -1:
            unvisited_squares = unvisited_squares[unvisited_squares['grid_id'] == grid_id]
        
        unvisited_square_features = torch.tensor(np.stack(unvisited_squares['features'].values)).float().to(self.device)
        train_x = torch.tensor(np.stack(train_x)).float().to(self.device)
        
        combined = torch.cat((train_x, unvisited_square_features))
        combined_mean = combined.mean(dim=0)
        combined_var = combined.var(dim=0)
        
        unvisited_square_features = (unvisited_square_features - combined_mean) / combined_var
        train_x = (train_x - combined_mean) / combined_var
        train_y = torch.tensor(train_y).float().to(self.device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().float()
        model = SingleTaskGP(train_x, train_y, likelihood).float()

        model = self._set_lm_parameters(model)
        model.eval().to(self.device)
        likelihood.eval().to(self.device)

        with torch.no_grad():
            sample = likelihood(model(unvisited_square_features))
            ucb_probs = upper_confidence_bound_v2(sample)
            unvisited_squares['GP_probs'] = ucb_probs

        return unvisited_squares

    
    def run_mm_gp(self, candidate_holes=None, hole_ids=None, square_ids=None, save_candidate_holes=False, active=False):
        # either run on all holes (all none) or candidate holes (you pass me the holes to run on)
        # or hole_ids (run only on these hole ids) or square ids (run on all unvisited holes with these square ids) TODO implement this

        # do the same thing as run_lm but for mm
        visited_holes = self.current_mm_state[(self.current_mm_state['visited'])].dropna(subset=['features', 'ctf', 'ice_thickness'])

        if len(visited_holes) > 1:
            train_x = torch.tensor(np.stack(visited_holes['features'].values)).float()
            train_y = torch.tensor(np.stack(visited_holes[['ice_thickness', 'ctf']].values).astype('float')).float()

            if len(self.historical_lm_state) > 0:
                historical_train_x = []
                historical_train_y = []
                for mm_state in self.historical_mm_state:
                    visited_holes = mm_state[(mm_state['visited'])].dropna(subset=['features', 'ctf', 'ice_thickness'])
                    historical_train_x.append(torch.tensor(np.stack(visited_holes['features'].values)).float())
                    historical_train_y.append(torch.tensor(np.stack(visited_holes[['ice_thickness', 'ctf']].values).astype('float')).float())

                if len(historical_train_x) > 0:
                    historical_train_x.append(train_x)
                    historical_train_y.append(train_y)
                    train_x = torch.cat(historical_train_x)
                    train_y = torch.cat(historical_train_y)

            if self.modeling_cap is not None:
                train_x = train_x[max(0, len(train_x) - self.modeling_cap):]
                train_y = train_y[max(0, len(train_y) - self.modeling_cap):]

            train_x = train_x.to(self.device)
            train_y = train_y.to(self.device)
           
            mm_modeling_time_start = time.time()
            
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2).float()
            model = MultitaskGP(train_x, train_y, likelihood, n_tasks=2).float()
            
            model = self._set_mm_parameters(model)
            model = model.to(self.device)
            likelihood = likelihood.to(self.device)
            # model = model.double()
            # likelihood = likelihood.double()

        # If candidate_holes is None, run the model on all unvisited holes
        # Else, add candidate holes to mm_state and only run model on candidate holes
        if candidate_holes:
            holes_to_run = candidate_holes

            if save_candidate_holes:
                self.current_mm_state = pd.concat((self.current_mm_state, candidate_holes))
        
        elif active:
            holes_to_run = self.current_mm_state.loc[list(self.active_holes)]
        elif hole_ids:
            holes_to_run = self.current_mm_state.loc[hole_ids]
        elif square_ids:
            holes_to_run = self.current_mm_state[self.current_mm_state['square_id'].isin(square_ids)]
        else:
            holes_to_run = self.current_mm_state[~self.current_mm_state['visited']]      
                
        unvisited_hole_features = torch.tensor(np.stack(holes_to_run['features'].values)).float().to(self.device)

        if len(visited_holes) > 1:
            model.eval()
            likelihood.eval()

            with torch.no_grad():
                sample = likelihood(model(unvisited_hole_features))
                holes_to_run['ctf_pred'] = sample.mean[:, 1]
                holes_to_run['ice_pred'] = sample.mean[:, 0]
                holes_to_run['ctf_var'] = sample.variance[:, 1]
                holes_to_run['ice_var'] = sample.variance[:, 0]

            mm_modeling_time_end = time.time()
            if mm_modeling_time_end - mm_modeling_time_start > self.settings['mm_modeling_time_soft_cap_seconds']:
                if self.modeling_cap is not None:
                    self.modeling_cap -= 5
                else:
                    self.modeling_cap = len(train_x)
                    print('hit modeling cap at {} seconds with {} train datapoints'.format(round(mm_modeling_time_end - mm_modeling_time_start), self.modeling_cap))

        else:
            holes_to_run['ctf_pred'] = 2.0
            holes_to_run['ice_pred'] = 50.0
            holes_to_run['ctf_var'] = 1.0
            holes_to_run['ice_var'] = 20.0

        return holes_to_run


    def add_hole_to_state(self, square_id, grid_id, mm_img_id, center, features, prior_score, radius=None, hole_id=None, visited=False, ctf=None, ice_thickness=None):
        if not hole_id:
            hole_id = len(self.current_mm_state)
        self.current_mm_state.loc[hole_id] = {'square_id': square_id, 'grid_id': grid_id, 'mm_img_id': mm_img_id, 'center_x': center[0], 'center_y':center[1], 'features': features, 'prior_score': prior_score, 'radius': radius,'visited': visited, 'ctf': ctf, 'ice_thickness': ice_thickness}
        return hole_id


    def add_square_to_state(self, grid_id, tile_id, center, features, prior_score, vertices, brightness, area, square_id=None, visited=False):
        if not square_id:
            square_id = len(self.current_lm_state)

        # check uniqueness of square id here

        self.current_lm_state.loc[square_id] = {'grid_id': grid_id, 'tile_id': tile_id, 'center_x': center[0], 'center_y': center[1], 'vert_1_x': vertices[0][0], 'vert_1_y': vertices[0][1], 'vert_2_x': vertices[1][0], 'vert_2_y': vertices[1][1], 'vert_3_x': vertices[2][0], 'vert_3_y': vertices[2][1], 'vert_4_x': vertices[3][0], 'vert_4_y': vertices[3][1], 'brightness': brightness, 'area': area, 'features': features, 'prior_score': prior_score, 'visited': visited}


    def visit_square(self, square_id):
        self.current_lm_state.loc[square_id, 'visited'] = True 
    

    def visit_hole(self, hole_id, ctf=None, ice_thickness=None, auto_mark_square_visited=True):
        self.current_mm_state.loc[hole_id, ['ctf', 'ice_thickness', 'visited']] = ctf, ice_thickness, True

        if auto_mark_square_visited:
            self.current_lm_state.loc[self.current_mm_state.loc[hole_id, 'square_id'], 'visited'] = True


    def mutate_lm_state(self, square_id, columnname, new_value):
        self.current_lm_state.loc[square_id, columnname] = new_value


    def mutate_mm_state(self, hole_id, columnname, new_value):
        self.current_mm_state.loc[hole_id, columnname] = new_value


    def _set_mm_parameters(self, model):
        model.load_state_dict(torch.load(self.settings['mm_gp_state_dict_path']))
        return model


    def set_active_holes(self, hole_ids):
        """
        Convenience method for defining a set of "currently active holes", from the most recent medium-mag image.

        run_mm_gp can be instructed to run on the active holes

        Visited holes are automatically removed from the active holes set. 
        """
        self.active_holes = set(hole_ids)

