import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import importlib as imp
import netCDF4 as nc
import os
import tensorflow as tf
import keras
from keras.models import load_model
import numpy.linalg as LA
import evaluate_functions as ef
import pandas as pd
import argparse

import sys
sys.path.insert(1, "/barnes-scratch/mafern/ForceSMIP/ForceSMIP/")
import experiments as exp

data_name = "Train4_Val4_CESM2_tos_tos"
model_name = "internal_feature"
complete_name = model_name + "_" + data_name
settings = exp.get_experiment(model_name)

filename = 'internal_linear_tos_multivar.npz'
datapath = "/barnes-scratch/mafern/ForceSMIP/ForceSMIP/exp_data/"
predpath = "/barnes-scratch/mafern/ForceSMIP/ForceSMIP/saved_predictions/"
savepath = "../barnes-scratch/mafern/ForceSMIP/ForceSMIP/evaluate/figures/"
savefig_name = "tos_internal"

VARIABLE = "psl"
TIER = 1

sample_n = 1 #used for functions that plot a single sample
PLOT_PRED_EXAMPLE = True
PLOT_AVG_ABS_DIFF = True

PATTERN_CORRELATION = False

SUB_YEAR_N_PERIOD = 44
SUB_PERIOD_TREND = False #"Sub-period trend (pattern correlation and RMSE)"
FULL_PERIOD_TREND = True #"Full-period trend (pattern correlation and RMSE)"

#"Monthly and annual time series of global means and means in latitude bands (correlation and RMSE)"
#"Grid point, monthly & annual time series (global-mean correlation and MSE)"
#"Indices: ENSO, East-West Pacific SST gradient, NAO, SAM, Southern Ocean SST, Arctic Amplification factor (in tas)"

EVALUATE_ENSO = False #SST
EVALUATE_EW_PAC_SST = False #incomplete
EVALUATE_NAO = False #SLP 
EVALUATE_SAM = False #SLP
EVALUATE_SOI = False #incomplete
EVALUATE_AA = False #incomplete

lat = np.linspace(-88.75, 88.75, 72)
lon = np.linspace(1.25, 358.8, 144)
lat_n = np.size(lat)
lon_n = np.size(lon)

lower = plt.cm.RdBu_r(np.linspace(0,.49, 49))
white = plt.cm.RdBu_r(np.ones(2)*0.5)
upper = plt.cm.RdBu_r(np.linspace(0.51, 1, 49))
colors = np.vstack((lower, white, upper))
tmap = matplotlib.colors.LinearSegmentedColormap.from_list('terrain_map_white', colors)
    
# Atr = All training
# Ftr = Forced training; truth for training
# Itr = Internal training
# Ava = All validation
# Fva = Forced validation
# Iva = Internal validation

def load_npz(filename):
    npzdat = np.load(filename)
    Atr, Ftr, Itr, Ava, Fva, Iva, Ate, Fte, Ite, Aev = (npzdat['Atr'], npzdat['Ftr'], npzdat['Itr'],
                                                        npzdat['Ava'], npzdat['Fva'], npzdat['Iva'],
                                                        npzdat['Ate'], npzdat['Fte'], npzdat['Ite'],
                                                        npzdat['Aev'])
    return Atr, Ftr, Itr, Ava, Fva, Iva, Ate, Fte, Ite, Aev

Atr, Ftr, Itr, Ava, Fva, Iva, Ate, Fte, Ite, Aev = load_npz(datapath + data_name + ".npz")
Atr = np.nan_to_num(Atr)
Ftr = np.nan_to_num(Ftr)
Itr = np.nan_to_num(Itr)
time = pd.date_range(start='1/1/1880', end = None, periods=73, freq = "Y")
time_n = np.size(time)

# Ptrin = prediction on training data
# Pval = prediciton on valuation
# Peval = predication on evaluation; truth not known.

f = np.load(predpath + complete_name + "0_preds.npz")
Ptrain, Pval, Peval, Ptest = f['Ptrain'], f['Pval'], f['Peval'], f['Ptest']
Ptrain = np.nan_to_num(Ptrain)

#######################################
n_members = 25
# set Truth variable and Prediction variable to the respective variable for the rest of the code to run
Truth = Fte.squeeze().reshape(n_members, 73, 72, 144)
Prediction = Ptest.reshape(n_members, 73, 72, 144)
Full = (Ite+Fte).squeeze().reshape(n_members, 73, 72, 144)
Truth = np.nan_to_num(Truth)
Prediction = np.nan_to_num(Prediction)
Full = np.nan_to_num(Full)

Prediction = xr.DataArray(Prediction, dims = ['members','time','lat','lon'])
Prediction["members"] = np.arange(n_members)
Prediction["time"] = time
Prediction["lat"] = lat[:]
Prediction["lon"] = lon[:]

Truth = xr.DataArray(Truth, dims = ['members','time','lat','lon'])
Truth["members"] = np.arange(n_members)
Truth["time"] = time
Truth["lat"] = lat[:]
Truth["lon"] = lon[:]

Full = xr.DataArray(Full, dims = ['members','time','lat','lon'])
Full["members"] = np.arange(n_members)
Full["time"] = time
Full["lat"] = lat[:]
Full["lon"] = lon[:]
#######################################
if EVALUATE_SAM:
    for member in range(n_members):
        Pred = Prediction[member]
        Tru = Truth[member]
        SAM_truth, eigenvectors = ef.CalcSAMIndex(lat, lon, Tru, None)
        SAM_pred, eigenvectors= ef.CalcSAMIndex(lat, lon, Pred, eigenvectors)
        
        plt.figure(figsize = (10,6))
        ef.PlotLines(time, SAM_pred, "time", "SAM index", "", "black", "Prediction", None, None)
        ef.PlotLines(time, SAM_truth, "time", "SAM index", "", "red", "Truth", None, None)
        plt.savefig(savepath + str(member) + "_SAM_" + savefig_name)

if EVALUATE_NAO:
    for member in range(n_members):
        Pred = Prediction[member]
        Tru = Truth[member]
        NAO_pred = ef.CalcNAOIndex(lat, lon, Pred)
        NAO_truth = ef.CalcNAOIndex(lat, lon, Tru)

        plt.figure(figsize = (10,6))
        ef.PlotLines(time, NAO_pred, "time", "NAO index", "", "black", "Prediction", None, None)
        ef.PlotLines(time, NAO_truth, "time", "NAO index", "", "red", "Truth", None, None)
        plt.savefig(savepath + str(member) + "_NAO_" + savefig_name)

if EVALUATE_ENSO:
    ENSO_pred = ef.CalcENSOIndex(lat, lon, Prediction)
    ENSO_truth = ef.CalcENSOIndex(lat, lon, Truth)
    
    plt.figure(figsize = (10,6))
    ef.PlotLines(time, ENSO_pred, "time", "ENSO index", "", "black", "Prediction", None, None)
    ef.PlotLines(time, ENSO_truth, "time", "ENSO index", "", "red", "Truth", None, None)
    plt.savefig(savepath + str(SAM) + "_" + savefig_name)

if FULL_PERIOD_TREND:
    # for member in range(n_members):
    for member in range(7):
        Pred = Prediction[member]
        Tru = Truth[member]
        Ful = Full[member]

        Truth_Trend = np.empty(shape = (72,144))
        Predicted_Trend = np.empty(shape = (72,144))
        Full_Trend = np.empty(shape = (72, 144))
        x_range = np.arange(0,time_n,1)

        for la in np.arange(0, lat_n, 1):
            for lo in np.arange(0, lon_n, 1):
                values = np.polyfit(x_range, Tru[:,la,lo], 1)
                Truth_Trend[la,lo] = values[0] * time_n

                values = np.polyfit(x_range, Pred[:,la,lo], 1)
                Predicted_Trend[la,lo] = values[0] * time_n

                values = np.polyfit(x_range, Ful[:,la,lo], 1)
                Full_Trend[la,lo] = values[0] * time_n

        # Plot_Gobal_Map(Truth_Trend, "Spatial trend of Truth", -.2, .2, tmap, "")
        # Plot_Gobal_Map(Predicted_Trend, "Spatial trend of Prediction", -.2, .2, tmap, "")
        # Plot_Gobal_Map(Predicted_Trend-Truth_Trend, "Difference between truth and prediction", -.2, .2, tmap, "")
        weighted_difference = (Predicted_Trend-Truth_Trend) * np.transpose(np.array([np.cos(np.deg2rad(lat)),]*144))
        RMSE = np.sum((weighted_difference)**2/time_n)
        PC = ef.CalcPatternCorrelation(Truth_Trend, Predicted_Trend)

        axs = ef.Plot2b2_colormesh_o3(Truth_Trend, Predicted_Trend, Predicted_Trend-Truth_Trend, Full_Trend, lat, lon, -2, 2)
        # axs.text(430, 8, "The weighted RMSE is: " +str(round(RMSE, 2)), fontsize = 10)
        # axs.text(430, 30, "The weighted Pattern Correlation is: " +str(round(PC)), fontsize = 10)

        plt.savefig("/barnes-scratch/mafern/ForceSMIP/ForceSMIP/evaluate/figures/" + str(member) + "_Full_trend_" + savefig_name)

if SUB_PERIOD_TREND:
    for member in range(n_members):
        Pred = Prediction[member]
        Tru = Truth[member]

        sub_time = np.arange(0, time_n, SUB_YEAR_N_PERIOD)
        x_range_sub = np.arange(0,SUB_YEAR_N_PERIOD,1)

        for s, sub in enumerate(sub_time[:-1]):
            Truth_Trend = np.empty(shape = (72,144))
            Predicted_Trend = np.empty(shape = (72,144))
            for la in np.arange(0, lat_n, 1):
                for lo in np.arange(0, lon_n, 1):
                    values = np.polyfit(x_range_sub, Tru[sub:sub_time[s+1],la,lo], 1)
                    Truth_Trend[la,lo] = values[0] * SUB_YEAR_N_PERIOD

                    values = np.polyfit(x_range_sub, Pred[sub:sub_time[s+1],la,lo], 1)
                    Predicted_Trend[la,lo] = values[0] * SUB_YEAR_N_PERIOD

            # Plot_Gobal_Map(Predicted_Trend-Truth_Trend, "Difference between truth and prediction", -1, 1, tmap, "")

            weighted_difference = (Predicted_Trend-Truth_Trend) * np.transpose(np.array([np.cos(np.deg2rad(lat)),]*144))
            RMSE = np.sum((weighted_difference)**2/SUB_YEAR_N_PERIOD)
            PC = ef.CalcPatternCorrelation(Truth_Trend[:,:], Predicted_Trend[:,:]) 

            axs = ef.Plot2b2_colormesh_o3(Truth_Trend, Predicted_Trend, Predicted_Trend-Truth_Trend, lat, lon, -.2, .2)
            axs.text(430, 8, "The weighted RMSE is: " +str(round(RMSE, 2)), fontsize = 10)
            axs.text(430, 30, "The weighted Pattern Correlation is: " +str(round(PC)), fontsize = 10)

            plt.savefig("/barnes-scratch/mafern/ForceSMIP/ForceSMIP/evaluate/figures/" + str(member) + "_Sub_trend_" + savefig_name)

if PLOT_PRED_EXAMPLE:
    ef.Plot_Gobal_Map(lat, lon, Prediction[sample_n,0,:,:], "Example of a Prediction", Prediction[sample_n,0,:,:].min(), Prediction[sample_n,0,:,:].max(), "Reds", "")
    plt.savefig(savepath + "sample_" + savefig_name)

if PLOT_AVG_ABS_DIFF:
    Difference = np.mean(np.abs(Prediction[sample_n,:,:,:] - Truth[sample_n,:,:,:]), axis = 0)
    # Plot_Gobal_Map(plot_data, title, min, max, colorbar, colorbar_title)
    ef.Plot_Gobal_Map(lat, lon, Difference, "Average Absolute Difference between \nTruth and Prediction", Difference.min(), Difference.max(), tmap, "")
    plt.savefig(savepath + "sample_difference_" + savefig_name)
    


