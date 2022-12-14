import json
import os

import numpy as np
import pandas as pd
import torch
import gpytorch


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


class Ptolemy_AL:
    """
    Class that manages running the active learning algorithms for Ptolemy to update scores on-the-fly. 

    Holds state for low-mag and medium-mag cases

    Initially we are going to use fixed GP hyperparameters.
    At some point should probably do some experiments that show this is better
    """

    def __init__(self, config='default', lm_state=None, mm_state=None):
        # we enforce lm_state to have "square_id" be the index
        if not lm_state:
            self.lm_state = pd.DataFrame(columns= ['square_id', 'grid_id', 'center', 'features', 'prior_score', 'visited'])
        else:
            self.lm_state = lm_state

        if self.lm_state.index.name != 'square_id':
            self.lm_state = self.lm_state.set_index('square_id')

        if not mm_state:
            self.mm_state = pd.DataFrame(columns= ['hole_id', 'square_id', 'grid_id', 'center', 'features', 'prior_score', 'visited', 'ctf', 'ice_thickness'])
        else:
            self.mm_state = mm_state

        if self.mm_state.index.name != 'hole_id':
            self.mm_state = self.mm_state.set_index('hole_id')

        if config == 'default':
            config_path = os.path.dirname(os.path.realpath(__file__)) + '/default_config.json'
        else:
            config_path = config
        
        self.load_config(config_path)

    def load_config(self, config_path):
        self.settings = json.load(open(config_path, 'r'))

    def load_lm_state(self, lm_state_path):
        """
        Loads and overwrites existing lm_state from reading lm_state_path dataframe pickle
        """
        self.lm_state = pd.read_pickle(lm_state_path)
        if self.lm_state.index.name != 'square_id':
            self.lm_state = self.lm_state.set_index('square_id')

    def load_mm_state(self, mm_state_path):
        """
        Loads and overwrites existing mm_state from reading mm_state_path dataframe pickle
        """
        self.mm_state = pd.read_pickle(mm_state_path)
        if self.mm_state.index.name != 'hole_id':
            self.mm_state = self.mm_state.set_index('hole_id')

    def compute_lengthscale(self):
        # Low mag lengthscale is computed just based on the square features alone. 
        # No visiting required.
        lengthscale = np.quantile(np.stack(self.lm_state.features.values), q=0.25, axis=0)
        return torch.tensor(lengthscale).unsqueeze(0)


    def _set_lm_parameters(self, model):
        model.mean_module.constant = torch.nn.Parameter(torch.tensor([float(self.settings['lm_gp_mean_constant'])])) # default should be 20 for now
        all_square_features = torch.tensor(self.lm_state[['features']].to_numpy())
        model.covar_module.base_kernel.raw_lengthscale = torch.nn.Parameter(torch.quantile(all_square_features, q=0.25, dim=0).unsqueeze(0))
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

        train_x, train_y = [], []

        visited_holes = self.mm_state[self.mm_state['visited']].dropna(subset=['features', 'ctf', 'ice_thickness'])
        visited_squares = self.lm_state[self.lm_state['visited']].dropna(subset=['features'])

        for square_id, row in visited_squares.iterrows():
            train_x.append(row.features)
            holes = visited_holes[visited_holes.square_id == square_id]
            counts = (holes.ctf < 5).sum()
            train_y.append(counts)

        train_x = torch.tensor(train_x)
        train_y = torch.tensor(train_y)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = SingleTaskGP(train_x, train_y, likelihood)

        model = self._set_lm_parameters(model)

        unvisited_squares = self.lm_state[~self.lm_state['visited']]

        unvisited_square_features = torch.tensor(unvisited_squares[['features']].to_numpy())

        model.eval()
        likelihood.eval()

        with torch.no_grad():
            sample = likelihood(model(unvisited_square_features))
            ucb_probs = upper_confidence_bound(sample)
            unvisited_squares['GP_probs'] = ucb_probs.numpy()

        if grid_id != -1:
            return unvisited_squares[unvisited_squares['grid_id'] == grid_id]
        else:
            return unvisited_squares

    
    def run_mm_gp(self, candidate_holes=None, hole_ids=None, square_ids=None, save_candidate_holes=False):
        # either run on all holes (all none) or candidate holes (you pass me the holes to run on)
        # or hole_ids (run only on these hole ids) or square ids (run on all unvisited holes with these square ids) TODO implement this

        # do the same thing as run_lm but for mm
        visited_holes = self.mm_state[(self.mm_state['visited'])].dropna(subset=['features', 'ctf', 'ice_thickness'])

        train_x = torch.tensor(visited_holes[['features']].to_numpy())
        train_y = torch.tensor(visited_holes[['ctf, ice_thickness']].to_numpy())

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = SingleTaskGP(train_x, train_y, likelihood)
        
        model = self._set_mm_parameters(model)

        # If candidate_holes is None, run the model on all unvisited holes
        # Else, add candidate holes to mm_state and only run model on candidate holes
        if candidate_holes:
            holes_to_run = candidate_holes

            if save_candidate_holes:
                self.mm_state = pd.concat((self.mm_state, candidate_holes))

        elif hole_ids:
            holes_to_run = self.mm_state.loc[holes_to_run]
        elif square_ids:
            holes_to_run = self.mm_state[self.mm_state['square_id'].isin(square_ids)]        
        else:
            holes_to_run = self.mm_state[~self.mm_state['visited']]
        
        unvisited_hole_features = torch.tensor(holes_to_run[['features']].to_numpy())

        model.eval()
        likelihood.eval()

        with torch.no_grad():
            sample = likelihood(model(unvisited_hole_features))
            holes_to_run['ctf_pred'] = sample.mean[:, 0]
            holes_to_run['ice_pred'] = sample.mean[:, 1]
            holes_to_run['ctf_var'] = sample.variance[:, 0]
            holes_to_run['ice_var'] = sample.variance[:, 1]

        return holes_to_run

    def add_hole_to_state(self, square_id, grid_id, center, features, prior_score, hole_id=None, visited=False, ctf=None, ice_thickness=None):
        if not hole_id:
            hole_id = len(self.mm_state)
        self.mm_state.loc[hole_id] = {'square_id': square_id, 'grid_id': grid_id, 'center': center, 'features': features, 'prior_score': prior_score, 'visited': visited, 'ctf': ctf, 'ice_thickness': ice_thickness}
        return hole_id

    def add_square_to_state(self, grid_id, center, features, prior_score, square_id=None, visited=False):
        if not square_id:
            square_id = len(self.lm_state)

        # check uniqueness of square id here

        self.lm_state.loc[square_id] = {'grid_id': grid_id, 'center': center, 'features': features, 'prior_score': prior_score, 'visited': visited}

    def visit_square(self, square_id):
        self.lm_state.loc[square_id, 'visited'] = True 
    
    def visit_hole(self, hole_id, ctf=None, ice_thickness=None, auto_mark_square_visited=True):
        self.mm_state.loc[hole_id, ['ctf', 'ice_thickness', 'visited']] = ctf, ice_thickness, True

        if auto_mark_square_visited:
            self.lm_state.loc[self.mm_state.loc[hole_id, 'square_id'], 'visited'] = True

    def mutate_lm_state(self, square_id, columnname, new_value):
        self.lm_state.loc[square_id, columnname] = new_value

    def mutate_mm_state(self, hole_id, columnname, new_value):
        self.mm_state.loc[hole_id, columnname] = new_value

    def _set_mm_parameters(self, model):
        model.load_state_dict(torch.load(self.settings['mm_gp_state_dict_path']))
        return model

