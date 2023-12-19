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
if os.path.isfile(settings['npz_dir'] + settings['exp_name'] + '.npz'):
    # Load the information, if it is already saved
    Atrain, Ftrain, Itrain, Aval, Fval, Ival, Atest, Ftest, Itest, Aeval = preprocessing.load_npz(settings)
else:
    # for each input variable, save/load processed training/validation/testing? data and split off the target variable
    for ii, var in enumerate(settings["input_variable"]):
        # load the training and validation sets for this variable, dimensions: [samples x lat x lon]
        Atr, Ftr, Itr = preprocessing.make_data(models=settings["train_models"], var=var,
                                        timecut=settings["time_range"], mems=settings["train_members"])
        Ava, Fva, Iva = preprocessing.make_data(models=settings["val_models"], var=var,
                                        timecut=settings["time_range"], mems=settings["val_members"])
        # include our own testing split, if models and members have been set
        if settings["test_members"] is not None and settings["test_models"] is not None:
                Ate, Fte, Ite = preprocessing.make_data(models=settings["test_models"], var=var,
                                        timecut=settings["time_range"], mems=settings["test_members"])
        else:
                # for compatibility, save something small as a placeholder
                Ate, Fte, Ite = [np.zeros([1, 1, 1])+np.nan] * 3

        # now, the evaluation data -- default to predicting for all 10 evaluation members
        for evalmem in evalmems:
            Aev_mem = preprocessing.make_eval_mem(evalmem=evalmem, var=var, timecut=settings["time_range"])
            if evalmem == '1A':
                Aev = Aev_mem
            else:
                Aev = np.vstack([Aev, Aev_mem])
        
        # put these into an array in the shape [samples, lat, lon, variable]
        if ii == 0:
            Atrain, Aval, Atest = Atr[..., None], Ava[..., None], Ate[..., None]
            Ftrain, Fval, Ftest = Ftr[..., None], Fva[..., None], Fte[..., None]
            Itrain, Ival, Itest = Itr[..., None], Iva[..., None], Ite[..., None]
            Aeval = Aev[..., None]
        else:
            Atrain = np.concatenate([Atrain, Atr[..., None]], axis=-1)
            Aval = np.concatenate([Aval, Ava[..., None]], axis=-1)
            Atest = np.concatenate([Atest, Ate[..., None]], axis=-1)
            Aeval = np.concatenate([Aeval, Aev[..., None]], axis=-1)
            Ftrain = np.concatenate([Ftrain, Ftr[..., None]], axis=-1)
            Fval = np.concatenate([Fval, Fva[..., None]], axis=-1)
            Ftest = np.concatenate([Ftest, Fte[..., None]], axis=-1)
            Itrain = np.concatenate([Itrain, Itr[..., None]], axis=-1)
            Ival = np.concatenate([Ival, Iva[..., None]], axis=-1)
            Itest = np.concatenate([Itest, Ite[..., None]], axis=-1)

    # use target_ind to get the target variable
    Ftrain = Ftrain[..., target_ind][..., None]
    Fval = Fval[..., target_ind][..., None]
    Ftest = Ftest[..., target_ind][..., None]
    Ival = Ival[..., target_ind][..., None]
    Itrain = Itrain[..., target_ind][..., None]
    Itest = Itest[..., target_ind][..., None]

    # save the preprocessed data             
    preprocessing.save_npz(settings, Atrain, Ftrain, Itrain, Aval, Fval, Ival, Atest, Ftest, Itest, Aeval)

    
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# standardize and select 'internal' or 'forced'
# first, using the evaluation data
Doer_eval = DataDoer(Atrain, Aval, Aeval, Ftrain, Fval, Itrain, Ival, settings)
Xtrain, Xval, Xeval, Ttrain, Tval = Doer_eval.do()
# second, using our own testing split, if requested (validation -> testing)
if settings["test_members"] is not None and settings["test_models"] is not None:
    Doer_test = DataDoer(Atrain, Atest, Aeval, Ftrain, Ftest, Itrain, Itest, settings)
    _, Xtest, _, _, Ttest = Doer_test.do()

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# build and train model
ved, encoder, decoder = VED.train_VED(Xtrain, Ttrain, Xval, Tval, settings)

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# save trained model
model_savename = os.path.join('saved_models', ARGS.exp_name, ARGS.exp_name+str(settings["seed"])+"_model")
tf.keras.models.save_model(ved, model_savename, overwrite=False)
# save weights
ved.save_weights(os.path.join('saved_models', ARGS.exp_name, ARGS.exp_name+str(settings["seed"])+"_weights.h5"))
# save training history
with open(os.path.join('saved_models', ARGS.exp_name, ARGS.exp_name+str(settings["seed"])+"_history.pickle"), "wb") as handle:
        pickle.dump(ved.history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# make predictions for the training, validation, and evaluation inputs
Ptrain, Pval, Peval = ved.predict(Xtrain), ved.predict(Xval), ved.predict(Xeval)
# convert back to unstandardized values
Ptrain_us, Pval_us, Peval_us = Doer_eval.unstandardize(Ptrain, Pval, Peval)

# if we have a testing split, predict that as well
if settings["test_members"] is not None and settings["test_models"] is not None:
     Ptest = ved.predict(Xtest)
     _, _, Ptest_us = Doer_test.unstandardize(Ptrain, Pval, Ptest)
else:
     if settings["target_component"] == 'internal':
          Ptest_us = Atest
     else:
          Ptest_us = np.zeros(np.shape(Atest)) + np.nan

# set the prediction output -- in both cases the forced response
# either directly, or from full minus internal
if settings["target_component"] == 'internal':
    Ptrain_out = Atrain - Ptrain_us
    Pval_out = Aval - Pval_us
    Peval_out = Aeval - Peval_us
    Ptest_out = Atest - Ptest_us
elif settings["target_component"] == 'forced':
    Ptrain_out = Ptrain_us
    Pval_out = Pval_us
    Peval_out = Peval_us
    Ptest_out = Ptest_us

# save the predictions
os.system('mkdir ' + settings['pred_dir'])
arr_name = settings['pred_dir'] + ARGS.exp_name+str(settings["seed"]) + "_preds.npz"
np.savez(arr_name,Ptrain=Ptrain_out[..., target_ind][..., None],
                  Pval=Pval_out[..., target_ind][..., None],
                  Peval=Peval_out[..., target_ind][..., None],
                  Ptest=Ptest_out[..., target_ind][..., None],
                  )
