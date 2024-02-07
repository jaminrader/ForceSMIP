import preprocessing
import os
import pickle
import numpy as np
import VED
import tensorflow as tf
from standardize import standardize_all_data, unstandardize_predictions
import metrics
import json

def preprocess(settings):
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

    return Atrain, Ftrain, Itrain, Aval, Fval, Ival, Atest, Ftest, Itest 

def prep_data_for_training(Atrain, Ftrain, Itrain, Aval, Fval, Ival, Atest, Ftest, Itest , settings):
    Aall = np.concatenate([Atrain, Aval, Atest])
    nanbool = np.isnan(Aall).any(axis=0)
    for D in [Atrain, Ftrain, Itrain, Aval, Fval, Ival, Atest, Ftest, Itest,]:
        D[:, nanbool] = np.nan

    # standardize the maps to prepare for training
    return standardize_all_data(Atrain, Aval, Atest, 
                                Ftrain, Fval, Ftest, 
                                Itrain, Ival, Itest, 
                                settings)
    
def train_and_predict(Xtrain_stand, Ttrain_stand, Xval_stand, Tval_stand, Xtest_stand, Ttest_stand,
                      Atrain_stand, Aval_stand, Atest_stand, 
                      Ftrain, Fval, Ftest, Itrain, Ival, Itest, 
                      fmean, fstd, imean, istd,
                      ARGS, settings):
    ved, encoder, decoder = VED.train_VED(Xtrain_stand, Ttrain_stand, Xval_stand, Tval_stand, settings)
    # save trained model
    model_savename = os.path.join('saved_models', ARGS.exp_name, ARGS.exp_name+str(settings["seed"])+"_model")
    tf.keras.models.save_model(ved, model_savename, overwrite=True)
    # save weights
    ved.save_weights(os.path.join('saved_models', ARGS.exp_name, ARGS.exp_name+str(settings["seed"])+"_weights.h5"))
    # save training history
    with open(os.path.join('saved_models', ARGS.exp_name, ARGS.exp_name+str(settings["seed"])+"_history.pickle"), "wb") as handle:
            pickle.dump(ved.history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
        
    return PFtrain, PFval, PFtest, PItrain, PIval, PItest

def calculate_metrics(PFtrain, PFval, PFtest, PItrain, PIval, PItest, Ftrain, Fval, Ftest, Itrain, Ival, Itest, verbose=False, settings_dict = None):
    # returns either a dictionary with the computed metrics or appends the results field into a dictionary supplied
    results = {}

    # make the weights
    lats = np.linspace(-90, 90, PFtrain.shape[1]+1)
    lats = [(ll+lh)/2 for ll, lh in zip(lats[:-1], lats[1:])]
    weights = np.cos(np.deg2rad(lats))[None, :, None, None] # lats is second dim
    # calculate mae, mse, R2 (weighted and unweighted)
    metric_names = ['MAE', 'wMAE', 'MSE', 'wMSE', 'R2', 'wR2']
    metric_funcs = [metrics.MAE, metrics.MAE, metrics.MSE, metrics.MSE, metrics.R2, metrics.R2]
    metric_weights = [None, weights, None, weights, None, weights]

    split_names = ['Ftrain', 'Fval', 'Ftest', 'Itrain', 'Ival', 'Itest',]
    truth_dats = [Ftrain, Fval, Ftest, Itrain, Ival, Itest,]
    pred_dats = [PFtrain, PFval, PFtest, PItrain, PIval, PItest,]

    for metric_name, metric_func, metric_weight in zip(metric_names, metric_funcs, metric_weights):
        for split_name, truth_dat, pred_dat in zip(split_names, truth_dats, pred_dats):
            metric_val = np.round(metric_func(truth_dat, pred_dat, weights=metric_weight), decimals=3)
            results[metric_name + '_' + split_name] = metric_val
            if verbose:
                print(metric_name + '_' + split_name + ': ', str(metric_val))

    if settings_dict != None:
        settings_dict['results'] = results.copy()
        return settings_dict
    else:
        return results

# def make_json_friendly(specs_orig):
#     specs = specs_orig.copy()
#     # Removes numpy objects from dictionary, and turns lists into strings
#     for imod in specs.keys():
#             if type(specs[imod][key]) == np.ndarray:
#                 specs[imod][key] = specs[imod][key].tolist()
#             if type(specs[imod][key]) == list:
#                 specs[imod][key] = str(specs[imod][key])
#             if type(specs[imod][key]) == np.int64:
#                 specs[imod][key] = int(specs[imod][key])
#     return specs

def make_json_friendly(specs_orig):
    specs = specs_orig.copy()
    # Removes numpy objects from dictionary, and turns lists into strings
    for key in specs:
            if type(specs[key]) == np.ndarray:
                specs[key] = specs[key].tolist()
            if type(specs[key]) == list:
                specs[key] = str(specs[key])
            if type(specs[key]) == np.int64:
                specs[key] = int(specs[key])
            # if key == 'results':
            #     for reskey in specs[key]:
            #         specs[key][reskey]
    return specs

def save_experiment_specs(exp_name, specs_dict, directory):
    # Save experiment specs
    os.system('mkdir ' + directory)
    with open(directory + exp_name + ".json", 'w') as fp:
        json.dump(make_json_friendly(specs_dict), fp)
    with open(directory + exp_name + ".p", 'wb') as fp:
        pickle.dump(specs_dict, fp)