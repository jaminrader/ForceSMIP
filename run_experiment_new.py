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
import random

# evaluation member labeling
evalmems = ['1A', '1B', '1C', '1D', '1E', '1F', '1G', '1H', '1I', '1J',]

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# set up parser (run via command line, e.g. python3 run_experiment.py exp_name)
PARSER = argparse.ArgumentParser()
PARSER.add_argument('exp_name', type=str)
ARGS = PARSER.parse_args()

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# get experiment settings
# assert np.isin(ARGS.exp_name, exp.experiments.keys()), "no experiment with the name" + ARGS.exp_name + ", current experiments:" + exp.experiments.keys()
settings = exp.get_experiment(ARGS.exp_name)

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# set up random seeds
tf.keras.backend.clear_session()
np.random.seed(settings["seed"])
random.seed(settings["seed"])
tf.random.set_seed(settings["seed"])

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# pre-process data

# index of target variable
target_ind = np.where(np.array(settings["input_variable"]) == settings["target_variable"])[0]

# Check to see if the saved data has already been output
if os.path.isfile(settings['npz_dir'] + settings['data_name'] + '.npz'):
    # Load the information, if it is already saved
    Atrain, Ftrain, Itrain, Aval, Fval, Ival, Atest, Ftest, Itest = preprocessing.load_npz(settings)
else:
    # for each input variable, save/load processed training/validation/testing? data and split off the target variable
    for ii, var in enumerate(settings["input_variable"]):
        # load the training and validation sets for this variable, dimensions: [samples x lat x lon]
        Atr, Ftr, Itr = preprocessing.make_data(models=settings["train_models"], var=var,
                                        timecut=settings["time_range"], mems=settings["train_members"])
        Ava, Fva, Iva = preprocessing.make_data(models=settings["val_models"], var=var,
                                        timecut=settings["time_range"], mems=settings["val_members"])
        # include our own testing split using ensembles for tuning purposes
        if settings["evaluate"] == False:
            Ate, Fte, Ite = preprocessing.make_data(models=settings["test_models"], var=var,
                                    timecut=settings["time_range"], mems=settings["test_members"])
        # if it's evaluation data, get the 'all' maps (A) but make the truths nans
        else:        
            evalmem = settings['evaluate']
            Ate = preprocessing.make_eval_mem(evalmem=evalmem, var=var, timecut=settings["time_range"])
            Fte = np.full_like(Ate, np.nan)
            Ite = np.full_like(Ate, np.nan)

        # put these into an array in the shape [samples, lat, lon, variable]
        if ii == 0:
            Atrain, Aval, Atest = Atr[..., None], Ava[..., None], Ate[..., None]
            Ftrain, Fval, Ftest = Ftr[..., None], Fva[..., None], Fte[..., None]
            Itrain, Ival, Itest = Itr[..., None], Iva[..., None], Ite[..., None]
        else:
            Atrain = np.concatenate([Atrain, Atr[..., None]], axis=-1)
            Aval = np.concatenate([Aval, Ava[..., None]], axis=-1)
            Atest = np.concatenate([Atest, Ate[..., None]], axis=-1)
            Ftrain = np.concatenate([Ftrain, Ftr[..., None]], axis=-1)
            Fval = np.concatenate([Fval, Fva[..., None]], axis=-1)
            Ftest = np.concatenate([Ftest, Fte[..., None]], axis=-1)
            Itrain = np.concatenate([Itrain, Itr[..., None]], axis=-1)
            Ival = np.concatenate([Ival, Iva[..., None]], axis=-1)
            Itest = np.concatenate([Itest, Ite[..., None]], axis=-1)

    # use target_ind to get the target variable
    Ftrain = Ftrain[..., target_ind]
    Fval = Fval[..., target_ind]
    Ftest = Ftest[..., target_ind]
    Ival = Ival[..., target_ind]
    Itrain = Itrain[..., target_ind]
    Itest = Itest[..., target_ind]

    # save the preprocessed data             
    preprocessing.save_npz(settings, Atrain, Ftrain, Itrain, Aval, Fval, Ival, Atest, Ftest, Itest)
    
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# Make sure nans in same spot for all
print('Atrain shape,', Atrain.shape)
print('Aval shape,', Aval.shape)
print('Atest shape,', Atest.shape)
Aall = np.concatenate([Atrain, Aval, Atest])
nanbool = np.isnan(Aall).any(axis=0)
for D in [Atrain, Ftrain, Itrain, Aval, Fval, Ival, Atest, Ftest, Itest,]:
    D[:, nanbool] = np.nan

# standardize the maps to prepare for training
Xtrain_stand, Xval_stand, Xtest_stand, \
Atrain_stand, Aval_stand, Atest_stand, \
Itrain_stand, Ival_stand, Itest_stand, \
Ftrain_stand, Fval_stand, Ftest_stand, \
Ttrain_stand, Tval_stand, Ttest_stand, \
amean, astd, imean, istd, fmean, fstd = standardize_all_data(Atrain, Aval, Atest, 
                                                            Ftrain, Fval, Ftest, 
                                                            Itrain, Ival, Itest, 
                                                            settings)

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# build and train model
ved, encoder, decoder = VED.train_VED(Xtrain_stand, Ttrain_stand, Xval_stand, Tval_stand, settings)

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# save trained model
model_savename = os.path.join('saved_models', ARGS.exp_name, ARGS.exp_name+str(settings["seed"])+"_model")
tf.keras.models.save_model(ved, model_savename, overwrite=True)
# save weights
ved.save_weights(os.path.join('saved_models', ARGS.exp_name, ARGS.exp_name+str(settings["seed"])+"_weights.h5"))
# save training history
with open(os.path.join('saved_models', ARGS.exp_name, ARGS.exp_name+str(settings["seed"])+"_history.pickle"), "wb") as handle:
        pickle.dump(ved.history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# make predictions for the training, validation, and evaluation inputs
Ptrain_stand, Pval_stand, Ptest_stand = ved.predict(Xtrain_stand), ved.predict(Xval_stand), ved.predict(Xtest_stand)

print('Ptrain_stand:', Ptrain_stand.shape)
print('Xtrain_stand:', Xtrain_stand.shape)

# convert back to unstandardized values , note: PF refers to the predicted forced response, 
# and PI to the predicted internal variablility
PFtrain, PFval, PFtest, \
PItrain, PIval, PItest, \
PFtrain_stand, PFval_stand, PFtest_stand, \
PItrain_stand, PIval_stand, PItest_stand = unstandardize_predictions(Atrain_stand, Aval_stand, Atest_stand,
                                                                Ptrain_stand, Pval_stand, Ptest_stand,
                                                                fmean, fstd, imean, istd,
                                                                settings)

if settings['save_predictions']:
    # save the predictions
    os.system('mkdir ' + settings['pred_dir'])
    arr_name = settings['pred_dir'] + ARGS.exp_name+str(settings["seed"]) + "_preds.npz"
    np.savez(arr_name,
            PFtrain=PFtrain,
            PFtest=PFtest,
            PItrain=PItrain,
            PItest=PItest,
            Ftrain=Ftrain,
            Ftest=Ftest,
            Itrain=Itrain,
            Itest=Itest,
            )