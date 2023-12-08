import xarray as xr
import os
import argparse
import pickle
import numpy as np
import experiments as exp
import VED
import tensorflow as tf
import preprocessing
from standardize import *

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# some paths -- may not be needed
# casper_data_path = '/glade/campaign/cgd/cas/asphilli/ForceSMIP'
# training_path = os.path.join(casper_data_path, 'Training')
# eval_path = os.path.join(casper_data_path, 'Evaluation-Tier1')
# eval_pr = os.path.join(eval_path, 'Amon', 'pr', 'pr_mon_1H.195001-202212.nc')
# eval_tos = os.path.join(eval_path, 'Omon', 'tos', 'tos_mon_1H.195001-202212.nc')

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# set up parser (run via command line, e.g. python run_experiment.py exp_name)
PARSER = argparse.ArgumentParser()
PARSER.add_argument('exp_name', type=str)
ARGS = PARSER.parse_args()

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# get experiment settings
# assert np.isin(ARGS.exp_name, exp.experiments.keys()), "no experiment with the name" + ARGS.exp_name + ", current experiments:" + exp.experiments.keys()
experiment_settings = exp.get_experiment(ARGS.exp_name)

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# set up random seed
tf.keras.backend.clear_session()
np.random.seed(experiment_settings["seed"])
# random.seed(experiment_settings["seed"])
tf.random.set_seed(experiment_settings["seed"])

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

##### pre-process data
# get random indices for training and validation
# rng_seed = np.random.default_rng(experiment_settings["seed"])
# total_members = np.arange(np.sum(experiment_settings["nmembers"])) # assuming this input is a list of train, val members
# rnd_inds = []
# for nmem in experiment_settings["nmembers"]:
#     rnd_inds.append(rng_seed.choice(total_members, nmem, replace=False))
#     total_members = np.setdiff1d(total_members, rnd_inds)

for ii, var in enumerate(experiment_settings["input_variable"]):
    # Check to see if the saved data has already been output
    if not os.path.isfile(experiment_settings['npz_dir'] + experiment_settings['exp_name'] + '.npz'):
        # load the training and validation sets for this variable
        At, Ft, It = preprocessing.make_data(models=experiment_settings["train_models"], var=var,
                                        timecut=experiment_settings["time_range"], mems=experiment_settings["train_members"])

        Av, Fv, Iv = preprocessing.make_data(models=experiment_settings["val_models"], var=var,
                                        timecut=experiment_settings["time_range"], mems=experiment_settings["val_members"])
        
        preprocessing.save_npz(experiment_settings, At, Ft, It, Av, Fv, Iv)
    
    # Load the information
    At, Ft, It, Av, Fv, Iv = preprocessing.load_npz(experiment_settings)

    # put these into an array in the shape [samples, lat, lon, variable]
    if ii == 0:
        Atrain = At[:, :, :, np.newaxis]
        Aval = Av[:, :, :, np.newaxis]
        Ftrain = Ft[:, :, :, np.newaxis]
        Fval = Fv[:, :, :, np.newaxis]
        Itrain = It[:, :, :, np.newaxis]
        Ival = Iv[:, :, :, np.newaxis]
    else:
        Atrain = np.concatenate([Atrain, At[:, :, :, np.newaxis]], axis=-1)
        Aval = np.concatenate([Aval, Av[:, :, :, np.newaxis]], axis=-1)
        Ftrain = np.concatenate([Ftrain, Ft[:, :, :, np.newaxis]], axis=-1)
        Fval = np.concatenate([Fval, Fv[:, :, :, np.newaxis]], axis=-1)
        Itrain = np.concatenate([Itrain, It[:, :, :, np.newaxis]], axis=-1)
        Ival = np.concatenate([Ival, Iv[:, :, :, np.newaxis]], axis=-1)
        
    if var == experiment_settings["target_variable"]:
        target_ind = ii
        
# get the target variable
Ftrain = Ftrain[:, :, :, target_ind][..., None]
Fval = Fval[:, :, :, target_ind][..., None]
Ival = Ival[:, :, :, target_ind][..., None]
Itrain = Itrain[:, :, :, target_ind][..., None]
    
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# get the correct evaluation data to predict
Atest = preprocessing.make_eval_mem(evalmem="1H",var=experiment_settings["target_variable"],timecut="Tier1")[..., None]

# standardize and select 'internal' or 'forced'
# jamin's datadoer here
Doer = DataDoer(Atrain, Aval, Atest, Ftrain, Fval, Itrain, Ival, experiment_settings)
Xtrain, Xval, Xtest, Ttrain, Tval = Doer.do()

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# build model
ved, encoder, decoder = VED.build_VED(Xtrain, Ttrain, experiment_settings)

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# train model
ved, encoder, decoder = VED.train_VED(Xtrain, Ttrain, Xval, Tval, experiment_settings)

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# save trained model
tf.keras.models.save_model(ved, os.path.join('saved_models', ARGS.exp_name, ARGS.exp_name+str(experiment_settings["seed"])+"_model"), overwrite=False)
ved.save_weights(os.path.join('saved_models', ARGS.exp_name, ARGS.exp_name+str(experiment_settings["seed"])+"_weights.h5"))
with open(os.path.join('saved_models', ARGS.exp_name, ARGS.exp_name+str(experiment_settings["seed"])+"_history.pickle"), "wb") as handle:
        pickle.dump(ved.history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# make predictions and save standardized and unstandardized
Ptrain = ved.predict(Xtrain)
Pval = ved.predict(Xval)
Ptest = ved.predict(Xtest)

Ptrain_us, Pval_us, Ptest_us = Doer.unstandardize(Ptrain, Pval, Ptest)

if experiment_settings["target_component"] == 'internal':
    Ptrain_out = Atrain-Ptrain_us
    Pval_out = Aval-Pval_us
    Ptest_out = Atest-Ptest_us
elif experiment_settings["target_component"] == 'forced':
    Ptrain_out = Ptrain_us
    Pval_out = Pval_us
    Ptest_out = Ptest_us     

os.system('mkdir ' + experiment_settings['pred_dir'])
arr_name = experiment_settings['pred_dir'] + ARGS.exp_name+str(experiment_settings["seed"])+"_preds.npz"
np.savez(arr_name,Ptrain=Ptrain_out,
                  Pval=Pval_out,
                  Ptest=Ptest_out,
                  )
