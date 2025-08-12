from paraview.simple import *
import sys
import os
sys.path.insert(0, "./")
import site ## Importing this module normally appends site-specific paths to the module search path and adds callables,

sys.real_prefix = sys.prefix
sys.prefix = "/soft/applications/conda/2024-04-29/mconda3"
prev_length = len(sys.path)
site.addsitedir("/soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages")
sys.path[:] = sys.path[prev_length:] + sys.path[0:prev_length]


sys.path.append('/lus/eagle/projects/multiphysics_aesp/azy4/mini_app/ml_models_test')
sys.path.append('/lus/grand/projects/visualization/azy4/catalyst_based_ml/Mini-Apps/autoencoder_code_copy')

import numpy as np

import autoencoder_insitu_train
from autoencoder_insitu_train import SCALEFACTOR

producer = TrivialProducer(registrationName="grid")

def catalyst_execute(info):
    global producer
    producer.UpdatePipeline()
    node = info.catalyst_params

    assert node.has_path("catalyst/state")
    assert node.has_path("catalyst/channels/grid")
    assert node["catalyst/channels/grid/data/fields"][0].name() == "velocity"
    assert node["catalyst/state/timestep"] == info.timestep
    
    fields = node["catalyst/channels/grid/data/fields"]
    grid = node["catalyst/channels/grid/data/coordsets/"]
    x_vel_values = np.expand_dims(np.array(fields["velocity/values"]), axis=-1)
    
    xcoords = np.expand_dims(np.array(grid["coords/values/x"]), axis = -1)
    ycoords = np.expand_dims(np.array(grid["coords/values/y"]), axis = -1)
    zcoords = np.expand_dims(np.array(grid["coords/values/z"]), axis = -1)
    numFluidPtsPyth = len(x_vel_values)
        
    scaled_vel_values = x_vel_values*SCALEFACTOR

    grid_and_vel = np.transpose(np.concatenate((xcoords, ycoords, zcoords, scaled_vel_values), axis=-1), (1, 0))

    autoencoder_insitu_train.main(grid_and_vel, info.cycle)
    