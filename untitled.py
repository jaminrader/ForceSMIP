import xarray as xr
import numpy as np
# import model build, train, metrics

import experiments as exp
# get_experiment(exp_name, base_exp_name=None, settings_overwrite=None) return settings
import preprocessing
# load_model(model,var,timecut="Tier1",ntrainmems=10) return all_ens
# make_X_data(models = ["CESM2","MIROC6","CanESM5"], var = "tos", timecut = "Tier1", nmems = 20) return X
# def make_Y_data(models = ["CESM2","MIROC6","CanESM5"], var = "tos", timecut = "Tier1", nmems = 20)



# print experiment names in the experiments dictionary
print(exp.experiments.keys())

# get the experiment settings for the requested experiment
expname = str(input("which experiment to run?: "))
assert np.isin(expname, exp.experiments.keys()), "no experiment with the name" + expname

# pre-process data


# set up random seed (unless Jamin does this in the model build or train)


# build model


# train model




