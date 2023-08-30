import xarray as xr
import numpy as np
import VED
# build_encoder(Xtrain, settings) return encoder, input_layer
# build_decoder(Xtrain, settings) return decoder
# build_VED(Xtrain, settings) return ved, encoder, decoder

import experiments as exp
# get_experiment(exp_name, base_exp_name=None, settings_overwrite=None) return settings
import preprocessing
# load_model(model,var,timecut="Tier1",ntrainmems=10) return all_ens
# make_X_data(models = ["CESM2","MIROC6","CanESM5"], var = "tos", timecut = "Tier1", nmems = 20) return X
# def make_Y_data(models = ["CESM2","MIROC6","CanESM5"], var = "tos", timecut = "Tier1", nmems = 20)

casper_data_path = '/glade/campaign/cgd/cas/asphilli/ForceSMIP'



# print experiment names in the experiments dictionary
print(exp.experiments.keys())

# get the experiment settings for the requested experiment
exp_name = str(input("which experiment to run?: "))
assert np.isin(exp_name, exp.experiments.keys()), "no experiment with the name" + expname
experiment_settings = get_experiment(exp_name)

# pre-process data


# set up random seed (unless Jamin does this in the model build or train)


# build model


# train model


# save trained model - make directory for this


# separate script for making predictions?

