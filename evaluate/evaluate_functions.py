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
import matplotlib.colors

lower = plt.cm.RdBu_r(np.linspace(0,.49, 49))
white = plt.cm.RdBu_r(np.ones(2)*0.5)
upper = plt.cm.RdBu_r(np.linspace(0.51, 1, 49))
colors = np.vstack((lower, white, upper))
tmap = matplotlib.colors.LinearSegmentedColormap.from_list('terrain_map_white', colors)

########################
#########PLOT###########
########################

def Plot2b2_colormesh_o3(data1, data2, data3, lat, lon, min, max):
    fig, axs = plt.subplots(2, 2)
    # ax=plt.axes(projection= ccrs.PlateCarree())
    data1, lonsr = add_cyclic_point(data1, coord=lon)
    data2, lonsr = add_cyclic_point(data2, coord=lon)
    data3, lonsr = add_cyclic_point(data3, coord=lon)
    # data4, lonsr = add_cyclic_point(data4, coord=lon)

    axs[0, 0].pcolormesh(lonsr, lat, data1, cmap = tmap, vmin = min, vmax = max)
    axs[0, 0].set_title('Truth')
    axs[0, 1].pcolormesh(lonsr, lat, data2, cmap = tmap, vmin = min, vmax = max)
    axs[0, 1].set_title('Predicted')
    cs = axs[1, 0].pcolormesh(lonsr, lat, data3, cmap = tmap, vmin = min, vmax = max)
    cbar = plt.colorbar(cs,shrink=0.7,orientation='horizontal',label='', format='%.1f')
    axs[1, 0].set_title('Difference')
    axs[1, 1].remove()

    return(axs[1,0])

def PlotLines(x, y, x_label, y_label, title, color, label, line_style, alpha):
    plt.axhline(0, color = 'black', linewidth=2, linestyle='--', alpha = .5)
    if len(x) == 0:
        return
    plt.plot(x, y, color = color, linewidth=3, label = label, linestyle=line_style, alpha = alpha)
    plt.title(title, pad=14)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.ylim(lim[y],lim[y])
    plt.xlim(min(x), max(x))


def Plot_Gobal_Map(lat, lon, plot_data, title, min, max, colorbar, colorbar_title):
    plt.figure(dpi = (200), figsize = (6,4))
    ax=plt.axes(projection= ccrs.PlateCarree())
    data, lonsr = add_cyclic_point(plot_data, coord=lon)
    
    cs = plt.pcolormesh(lonsr, lat, data, 
        cmap = colorbar, vmin = min, vmax = max)
    ax.coastlines()
    cbar = plt.colorbar(cs,shrink=0.7,orientation='horizontal',label='Surface Air Temperature (K)', format='%.1f')#, pad=5)
    cbar.set_label(colorbar_title)
    plt.title(title)
    
    plt.show()

########################
#########CALC###########
########################
def CalcPatternCorrelation(y_true, y_pred):  
    ss_res = np.sum(np.square(y_true-y_pred))
    ss_tot = np.sum(np.square(y_true-np.mean(y_true)))
    return ( 1 - ss_res/(ss_tot + 10**-16) ) 

def nandot(X,Y):
    C = np.empty([np.size(X,axis=0),np.size(Y,axis=1)])
    for row in np.arange(0,np.size(X,axis=0)):
        for col in np.arange(0,np.size(Y,axis=1)):
            C[row,col] = np.nanmean(np.multiply(X[row,:],Y[:,col]))
    return C   

def CalcSAMIndex(lat, lon, data):
    print("Calculating SAM index")
    print("Not finished; need to test with SLP model")
    climatology = data.groupby("time.month").mean("time")
    data = data.groupby("time.month") - climatology

    #Selects the region used to calculate the SAM index
    lat = xr.DataArray(lat, dims = ['lat'], coords = dict(lat=lat))
    data_temp = data
    data = data.sel(lat=slice(-90, -20))
    lat_region = lat.sel(lat=slice(-90, -20))
    
    data = np.array(data)/100   
    dataf = data.reshape(len(data[:,0,0]), len(data[0,:,0]) * len(data[0,0,:]))  
    
    #Calculate EOF
    C = nandot(dataf, np.transpose(dataf))   
    lam, Z = LA.eig(C)
    Z = (Z - np.nanmean(Z,axis=0))/np.nanstd(Z,axis=0)
    E = np.dot(Z.T,dataf)
    
    print(np.shape(Z))
    print(np.shape(E))
    
    # D = nandot(Z[:,:10].T,data_temp.reshape(data.shape[0],data_temp.shape[1]*data_temp.shape[2]))
    # xplot = D.reshape(D.shape[0],len(data[0,:,0]),len(data[0,0,:]))[0,:,:]

    # plt.title("before fix")    
    # cs = plt.contourf(lon, lat, xplot, cmap = tmap)#, levels = np.linspace(-15, 15, 20))
    # plt.plot(52, -80 , 'o', color = 'red')
    # plt.plot(52, -40, 'o', color = 'blue')
    # plt.colorbar(cs)
    # plt.show()
    
    # #This checks that negative EOFs represent the negative phase of the SAM
    # if xplot[50, -80] <= 0 and xplot[20, -40] >=0:
    #     print("fix")
    #     Z = Z * -1
    #     xplot = xplot * -1
         
    # cs = plt.contourf(lon, lat_region, xplot, cmap = tmap, levels = np.linspace(-15, 15, 20))
    # plt.colorbar(cs)
    # plt.show()

    return(Z[:,0])

def CalcNAOIndex(lat, lon, data):
    print("Calculating NAO index")
    print("Not finished; need to test with SLP model")
    climatology = data.groupby("time.month").mean("time")
    data = data.groupby("time.month") - climatology

    #Selects the region used to calculate the NAO index
    data1 = data.sel(lat=slice(20, 80), lon=slice(-90, 40))
    data2 = data.sel(lat=slice(20, 80), lon=slice(270, 360))

    #Combines the two NAO regions
    data = xr.concat([data2, data1], dim="lon")

    data = np.array(data)/100 
    dataf = data.reshape(len(data[:,0,0]), len(data[0,:,0]) * len(data[0,0,:]))       

    #Calculate EOF
    C = nandot(dataf, np.transpose(dataf))
    lam, Z = LA.eig(C)
    Z = (Z - np.nanmean(Z,axis=0))/np.nanstd(Z,axis=0)
    E = np.dot(Z.T,dataf)
    
    # D = nandot(Z[:,:10].T,data.reshape(data.shape[0],data.shape[1]*data.shape[2]))
    # xplot = D.reshape(D.shape[0],len(data[0,:,0]),len(data[0,0,:]))[0,:,:]

    #This checks that negative EOFs represent the negative phase of the NAO
    # if xplot[50, 52] >= 0 and xplot[20, 52] <=0:
    #     print("fix")
    #     Z = Z * -1
    #     xplot = xplot * -1
            
    # cs = plt.contourf(xplot, cmap = tmap, levels = np.linspace(-5, 5, 20))
    # plt.colorbar(cs)
    # plt.plot(52, 20, 'ro', color = 'red')
    # plt.plot(52, 50, 'ro', color = 'blue')
    # plt.show()
    
    return(Z[:,0])


def CalcENSOIndex(lat, lon, data):
    print("Calculating ENSO index")
    lat = xr.DataArray(lat, dims = ['lat'], coords = dict(lat=lat))
    lon = xr.DataArray(lon, dims = ['lon'], coords = dict(lon=lon))
    climatology = data.groupby("time.month").mean("time")
    data = data.groupby("time.month") - climatology

    #standardized 
    data = data.groupby("time.month") / data.groupby("time.month").std("time")
    
    #Select the region used to caluclate ENSO index
    tos_nino34 = data.sel(lat=slice(-5, 5), lon=slice(190, 240))
    
    #Calculated weighted average
    weights = np.cos(np.deg2rad(tos_nino34.lat))
    weights.name = "weights"
    index_nino34 = tos_nino34.weighted(weights).mean(("lat", "lon"))
    
    #complete the running mean
    index_nino34 = index_nino34.rolling(time=5, center=True).mean()

    return(index_nino34)