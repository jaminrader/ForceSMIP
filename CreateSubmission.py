import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import netCDF4 as nc
import pandas as pd
import os
import sys
import argparse
sys.path.insert(1, "evaluate/")
import evaluate_functions as ef

# -------------------------------------------------------------------------------------------
# set up parser (run via command line, e.g. python3 CreateSubmission.py exp_name var tier)
# e.g., python3 CreateSubmission.py 'internal_test_standard' 'tos' 'T2'
PARSER = argparse.ArgumentParser()
PARSER.add_argument('exp_name', type=str)
PARSER.add_argument('var', type=str)
PARSER.add_argument('tier', type=str)
ARGS = PARSER.parse_args()
experiment_name = ARGS.exp_name
variable = ARGS.var # tos, tas, pr, psl, zmta, monmaxtasmax, monmintasmin, monmaxpr, mrso
Tier = ARGS.tier
# -------------------------------------------------------------------------------------------
if Tier == "T1":
    date = pd.date_range(start='1/1/1950', end='1/1/2023', freq='M')
    memberID = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
if Tier == "T2":
    date = pd.date_range(start='1/1/1900', end='1/1/2024', freq='M')
    memberID = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U"]
if Tier == "T3":
    date = pd.date_range(start='1/1/1979', end='1/1/2024', freq='M')
    memberID = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U"]
nyears = int(len(date)/12)

lat = np.linspace(-88.75, 88.75, 72) 
lat_n = np.size(lat)
lon = np.linspace(1.25, 358.8, 144)
lon_n = np.size(lon)
plev = np.linspace(1000, 200, 17)
plev_n = np.size(plev)
months = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "100", "110", "120"]
if variable == "zmta":
    IS_Zonal_avg = True
    shape = [plev_n, lat_n]
else:
    IS_Zonal_avg = False
    shape = [lat_n, lon_n]

DirectoryPath = "./"
loadDirectoryPath = DirectoryPath + "saved_predictions/"
saveDirectoryPath = DirectoryPath + "SubmissionFiles/"

CreateSaveFile = True
PlotForcedTrend = True
PlotExample, sample_n = True, 10
CheckSavedFile = True



predictions = np.empty(shape=(12, nyears*np.size(memberID), shape[0], shape[1], 1))
# for i, month in enumerate(months):
for i, month in enumerate(months):
    predictionFile = experiment_name + '_' + variable + "_" + month + "_preds.npz"
    f = np.load(loadDirectoryPath + predictionFile)
    PFtest = f['PFtest'] #(years*evaluation_members, lat, lon, 1)
    
    # used to mask to the prediction with nans
    Ftrain = f["Ftrain"][:nyears*np.size(memberID), :, :, :]
    PFtest[np.isnan(Ftrain)] = np.nan

    predictions[i, :, :, :, :] = PFtest

final_data = np.empty(shape=(np.size(memberID), nyears*12, shape[0], shape[1]))
for i, member in enumerate(memberID):
    data_member = predictions[:, (nyears*i):(nyears*(i+1)), :, :, 0]
    n = 0
    for t in range(0, nyears):
        final_data[i, n, :, :] = data_member[0, t, :, :]
        n = n + 1
        final_data[i, n, :, :] = data_member[1, t, :, :]
        n = n + 1
        final_data[i, n, :, :] = data_member[2, t, :, :]
        n = n + 1
        final_data[i, n, :, :] = data_member[3, t, :, :]
        n = n + 1
        final_data[i, n, :, :] = data_member[4, t, :, :]
        n = n + 1
        final_data[i, n, :, :] = data_member[5, t, :, :]
        n = n + 1
        final_data[i, n, :, :] = data_member[6, t, :, :]
        n = n + 1
        final_data[i, n, :, :] = data_member[7, t, :, :]
        n = n + 1
        final_data[i, n, :, :] = data_member[8, t, :, :]
        n = n + 1
        final_data[i, n, :, :] = data_member[9, t, :, :]
        n = n + 1
        final_data[i, n, :, :] = data_member[10, t, :, :]
        n = n + 1
        final_data[i, n, :, :] = data_member[11, t, :, :]
        n = n + 1

###############################
###############################
###############################
for i, member in enumerate(memberID):
    saveFilename = variable + "_" + memberID[i] + "_" + Tier + "_EncoderDecoderInternal_TPG"

    if CreateSaveFile:
        if IS_Zonal_avg:
            save_data = xr.DataArray(final_data[i, :, :, :], dims = ['time', 'plev', 'lat'],
                                coords = [date, plev, lat])
            save_data.name = variable

            exists = os.path.isfile(saveDirectoryPath + saveFilename + ".nc")
            if exists:
                os.remove(saveDirectoryPath + saveFilename + ".nc")

            save_data.to_netcdf(saveDirectoryPath + saveFilename + ".nc")
        else:
            save_data = xr.DataArray(final_data[i, :, :, :], dims=['time', 'lat', 'lon'], coords=[date, lat, lon])
            save_data.name = variable
            
            exists = os.path.isfile(saveDirectoryPath + saveFilename + ".nc")
            if exists:
                os.remove(saveDirectoryPath + saveFilename + ".nc")

            save_data.to_netcdf(saveDirectoryPath + saveFilename + ".nc")

###############################
###############################
###############################

    if PlotForcedTrend:
        if IS_Zonal_avg:
            globalmean = xr.DataArray(final_data[i, :, :, :], dims = ['time', 'plev', 'lat'])
            globalmean["time"] = date
            globalmean["lat"] = lat[:]
            globalmean["plev"] = plev[:]

            globalmean = ef.CalcGlobalMean(globalmean, lat)

            plt.figure()
            plt.plot(date, globalmean, label=memberID[i])
            plt.legend()
            plt.title("Predicted Forced Response for " + variable + " " + memberID[i])
            plt.savefig(DirectoryPath + "evaluate/figures/predictedForcedResponses" + memberID[i] + f"_{variable}{Tier}.png")
        else:
            globalmean = xr.DataArray(final_data[i, :, :, :], dims = ['time', 'lat', 'lon'])
            globalmean["time"] = date
            globalmean["lat"] = lat[:]
            globalmean["lon"] = lon[:]

            globalmean = ef.CalcGlobalMean(globalmean, lat)

            plt.figure()
            plt.plot(date, globalmean, label=memberID[i])
            plt.legend()
            plt.title("Predicted Forced Response for " + variable + " " + memberID[i])
            plt.savefig(DirectoryPath + "evaluate/figures/predictedForcedResponses" + memberID[i] + f"_{variable}{Tier}.png")

    if PlotExample:
        if IS_Zonal_avg:
            plt.figure(dpi=(200), figsize=(6, 4))
            cs = plt.pcolormesh(plev, lat, final_data[0, sample_n, :, :], cmap="Reds")
            cbar = plt.colorbar(cs, shrink=0.7, orientation='horizontal', label='Surface Air Temperature (C)', format='%.1f')#, pad=5)
            plt.title("Sample " + str(sample_n))
            plt.savefig(DirectoryPath + "evaluate/figures/sample" + memberID[i] + f"_{variable}{Tier}.png")
        else:
            plt.figure(dpi=(200), figsize=(6, 4))
            ax=plt.axes(projection= ccrs.PlateCarree())
            data, lonsr = add_cyclic_point(final_data[0, sample_n, :, :], coord=lon)
            cs = plt.pcolormesh(lonsr, lat, data, cmap="Reds")
            ax.coastlines()
            cbar = plt.colorbar(cs, shrink=0.7, orientation='horizontal',label='Surface Air Temperature (C)', format='%.1f')#, pad=5)
            # cbar.set_label(colorbar_title)
            plt.title("Sample " + str(sample_n))
            plt.savefig(DirectoryPath + "evaluate/figures/sample" + memberID[i] + f"_{variable}{Tier}.png")

    if CheckSavedFile:
        if IS_Zonal_avg:
            f = xr.open_dataset(saveDirectoryPath + saveFilename + ".nc")
            plotdata = f[variable]
            
            plt.figure(dpi=(200), figsize=(6, 4))
            cs = plt.pcolormesh(plotdata[0, :, :], cmap="Reds")
            cbar = plt.colorbar(cs)
            plt.title(variable + " " + memberID[i])
            plt.savefig(DirectoryPath + "evaluate/figures/OPEN" + memberID[i] + f"_{variable}{Tier}.png")
        else:
            f = xr.open_dataset(saveDirectoryPath + saveFilename + ".nc")
            plotdata = f[variable]
            
            plt.figure(dpi=(200), figsize=(6, 4))
            cs = plt.pcolormesh(lon, lat, plotdata[0, :, :], cmap ="Reds")
            cbar = plt.colorbar(cs)
            plt.title(variable + " " + memberID[i])
            plt.savefig(DirectoryPath + "evaluate/figures/OPEN" + memberID[i] + f"_{variable}{Tier}.png")
