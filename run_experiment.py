import xarray as xr
import os
import argparse
import pickle
import numpy as np
import experiments as exp
import VED
import preprocessing

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# some paths, may not be needed
casper_data_path = '/glade/campaign/cgd/cas/asphilli/ForceSMIP'
training_path = os.path.join(casper_data_path, 'Training')
eval_path = os.path.join(casper_data_path, 'Evaluation-Tier1')
eval_pr = os.path.join(eval_path, 'Amon', 'pr', 'pr_mon_1H.195001-202212.nc')
eval_tos = os.path.join(eval_path, 'Omon', 'tos', 'tos_mon_1H.195001-202212.nc')

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# set up parser (run via command line, e.g. python run_experiment.py exp_name)
PARSER = argparse.ArgumentParser()
PARSER.add_argument('exp_name', type=str)
ARGS = PARSER.parse_args()

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# get experiment settings
assert np.isin(ARGS.exp_name, exp.experiments.keys()), "no experiment with the name" + expname + ", current experiments:" + exp.experiments.keys()
experiment_settings = get_experiment(ARGS.exp_name)

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# set up random seed
tf.keras.backend.clear_session()
np.random.seed(experiment_settings["seed"])
random.seed(experiment_settings["seed"])
tf.random.set_seed(experiment_settings["seed"])

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------


##### pre-process data
# get random indices for training and validation
rng_seed = np.random.default_rng(experiment_settings["seed"])
total_members = np.arange(np.sum(experiment_settings["nmembers"])) # assuming this input is a list of train, val members
rnd_inds = []
for nmem in experiment_settings["nmembers"]:
    rnd_inds.append(rng_seed.choice(total_members, nmem, replace=False))
    total_members = np.setdiff1d(total_members, rnd_inds)

#### would need to change this to take in the indices of the members rather than the number of members
Xtrain = preprocessing.make_X_data(models=experiment_settings["models"], var=experiment_settings["variable"],
                                  timecut=experiment_settings["time_range"], nmems=rnd_inds[0])
Ytrain = preprocessing.make_Y_data(models=experiment_settings["models"], var=experiment_settings["variable"],
                                  timecut=experiment_settings["time_range"], nmems=rnd_inds[0])

# do we want a separate list of models for the validation?
Xval = preprocessing.make_X_data(models=experiment_settings["models"], var=experiment_settings["variable"],
                                  timecut=experiment_settings["time_range"], nmems=rnd_inds[1])
Yval = preprocessing.make_Y_data(models=experiment_settings["models"], var=experiment_settings["variable"],
                                  timecut=experiment_settings["time_range"], nmems=rnd_inds[1])

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# build model
ved, encoder, decoder = VED.build_VED(Xtrain, experiment_settings)

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# train model


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# save trained model
tf.keras.models.save_model(ved, os.path.join('saved_models', exp_name, exp_name+str(experiment_settings["seed"])+"_model"), overwrite=False)
ved.save_weights(os.path.join('saved_models', exp_name, exp_name+str(experiment_settings["seed"])+"_weights.h5"))
with open(os.path.join('saved_models', exp_name, exp_name+str(experiment_settings["seed"])+"_history.pickle"), "wb") as handle:
        pickle.dump(ved.history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# separate script for making predictions?


