import numpy as np
import xarray as xr
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


datapath = "/Users/charlotteconnolly/Desktop/ForceSMIP_code/"
lat = np.linspace(-90, 90, 72)
lon = np.linspace(0, 360, 144)
    
# Atr = All training
# Ftr = Forced training; truth for training
# Itr = Internal training
# Ava = All validation
# Fva = Forced validation
# Iva = Internal validation

def load_npz(filename):
    npzdat = np.load(filename)
    Atr, Ftr, Itr, Ava, Fva, Iva = npzdat['Atr'], npzdat['Ftr'], npzdat['Itr'], npzdat['Ava'], npzdat['Fva'], npzdat['Iva']
    return Atr, Ftr, Itr, Ava, Fva, Iva

Atr, Ftr, Itr, Ava, Fva, Iva = load_npz(datapath + "internal_linear_cesm2_tos_multivar.npz")
Atr = np.nan_to_num(Atr)
Ftr = np.nan_to_num(Ftr)
Itr = np.nan_to_num(Itr)
time = pd.date_range(start='1/1/2000', end = None, periods=len(Atr), freq = "M")

# Ptrin = prediction on training data
# Pval = prediciton on valuation
# Peval = predication on evaluation; truth not known.

f = np.load(datapath + "internal_linear_cesm2_tos_multivar0_preds.npz")
Ptrain, Pval, Peval = f['Ptrain'], f['Pval'], f['Peval']
Ptrain = np.nan_to_num(Ptrain)

#######################################
# set Truth variable and Prediction variable to the respective variable for the rest of the code to run
Truth = Ftr
Prediction = Ptrain[:,:,:,0]
Truth = np.nan_to_num(Truth)
Prediction = np.nan_to_num(Prediction)
#######################################

VARIABLE = "tos"
TIER = 1

PATTERN_CORRELATION = True

SUB_YEAR_N_PERIOD = 73
SUB_PERIOD_TREND = True #"Sub-period trend (pattern correlation and RMSE)"
FULL_PERIOD_TREND = False #"Full-period trend (pattern correlation and RMSE)"

#"Monthly and annual time series of global means and means in latitude bands (correlation and RMSE)"
#"Grid point, monthly & annual time series (global-mean correlation and MSE)"
#"Indices: ENSO, East-West Pacific SST gradient, NAO, SAM, Southern Ocean SST, Arctic Amplification factor (in tas)"

#Skewed becuase of all the zeros
if PATTERN_CORRELATION:
    PC = CalcPatternCorrelation(Truth[0,:,:], Prediction[0,:,:])
    print("The pattern correlation between the Truth and the Prediction is " + str(PC))

    PC = CalcPatternCorrelation(Atr[0,:,:], Ftr[0,:,:])
    print("The pattern correlation between complete data and the Forced component is " + str(PC))

    PC = CalcPatternCorrelation(Atr[0,:,:] - Prediction[0,:,:], Itr[0,:,:])
    print("The pattern correlation between the complete data minus the prediction and the internal variability is " + str(PC))

if FULL_PERIOD_TREND:
    Truth_Trend = np.empty(shape = (72,144))
    Predicted_Trend = np.empty(shape = (72,144))
    x_range = np.arange(0,time_n,1)

    for la in np.arange(0, lat_n, 1):
        for lo in np.arange(0, lon_n, 1):
            values = np.polyfit(x_range, Truth[:,la,lo], 1)
            Truth_Trend[la,lo] = values[0] * time_n

            values = np.polyfit(x_range, Prediction[:,la,lo], 1)
            Predicted_Trend[la,lo] = values[0] * time_n

    weighted_difference = (Predicted_Trend-Truth_Trend) * np.transpose(np.array([np.cos(np.deg2rad(lats)),]*144))
    
    RMSE = np.sum((weighted_difference)**2/time_n)
    print("For the full period trend, the weighted RMSE is: " +str(RMSE))

    PC = CalcPatternCorrelation(Truth_Trend, Predicted_Trend)
    print("For the full period trend, the weighted Pattern Correlation is: " +str(PC))

if SUB_YEAR_N_PERIOD:
    sub_time = np.arange(0, time_n, SUB_YEAR_N_PERIOD)
    x_range_sub = np.arange(0,SUB_YEAR_N_PERIOD,1)

    for s, sub in enumerate(sub_time[:-1]):
        Truth_Trend = np.empty(shape = (72,144))
        Predicted_Trend = np.empty(shape = (72,144))
        for la in np.arange(0, lat_n, 1):
            for lo in np.arange(0, lon_n, 1):
                values = np.polyfit(x_range_sub, Truth[sub:sub_time[s+1],la,lo], 1)
                Truth_Trend[la,lo] = values[0] * SUB_YEAR_N_PERIOD

                values = np.polyfit(x_range_sub, Prediction[sub:sub_time[s+1],la,lo], 1)
                Predicted_Trend[la,lo] = values[0] * SUB_YEAR_N_PERIOD

        weighted_difference = (Predicted_Trend-Truth_Trend) * np.transpose(np.array([np.cos(np.deg2rad(lats)),]*144))

        RMSE = np.sum((weighted_difference)**2/SUB_YEAR_N_PERIOD)
        print("For the sub trend, the RMSE is: " +str(RMSE))

        PC = CalcPatternCorrelation(Truth_Trend[:,:], Predicted_Trend[:,:])
        print("For the sub trend, the weighted Pattern Correlation is: " +str(PC))    


