import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import netCDF4 as nc
import os
lat = np.linspace(-88.75, 88.75, 72) 
lat_n = np.size(lat)
lon = np.linspace(1.25, 358.8, 144)
lon_n = np.size(lon)
plev = np.linspace(1000, 200, 37) #Incorrect
plev_n = np.size(plev)

months = ["10","20","30","40","50","60",
          "70","80","90","100","110","120"]
memberID = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
component = "forced"
variable = "zmta"#"monmintasmin"

loadDirectoryPath = "/barnes-scratch/mafern/ForceSMIP/ForceSMIP/saved_predictions/"
saveDirectoryPath = "/barnes-scratch/mafern/ForceSMIP/ForceSMIP/SubmissionFiles/"
Tier = "T1"
IS_Zonal_avg = False

PlotExample, sample_n = False, 10
CreateSaveFile = True
CheckSavedFile = False #not complete

predictions = np.empty(shape = (12,730,72,144,1))
for i, month in enumerate(months):
    predictionFile = component + "_test_standard_" + variable + "_" + month + "_preds.npz"
    f = np.load(loadDirectoryPath + predictionFile)
    PFtest = f['PFtest'] #(730, 72, 144, 1)

    # print(np.shape(PFtest))
    # cs = plt.contourf(PFtest[0,:,:,0])
    # plt.colorbar(cs)
    # plt.title("Opened data annual " + variable)
    # plt.savefig("/barnes-scratch/mafern/ForceSMIP/ForceSMIP/evaluate/figures/tempfig.png")

    predictions[i,:,:,:,:] = PFtest

final_data = np.empty(shape = (10,876,72,144))
for i, member in enumerate(memberID):
    data_member = predictions[:,(73*i):(73*(i+1)),:,:,0]

    n = 0
    for t in range(0, 73):
        final_data[i,n,:,:] = data_member[0,t,:,:]
        n = n + 1
        final_data[i,n,:,:] = data_member[1,t,:,:]
        n = n + 1
        final_data[i,n,:,:] = data_member[2,t,:,:]
        n = n + 1
        final_data[i,n,:,:] = data_member[3,t,:,:]
        n = n + 1
        final_data[i,n,:,:] = data_member[4,t,:,:]
        n = n + 1
        final_data[i,n,:,:] = data_member[5,t,:,:]
        n = n + 1
        final_data[i,n,:,:] = data_member[6,t,:,:]
        n = n + 1
        final_data[i,n,:,:] = data_member[7,t,:,:]
        n = n + 1
        final_data[i,n,:,:] = data_member[8,t,:,:]
        n = n + 1
        final_data[i,n,:,:] = data_member[9,t,:,:]
        n = n + 1
        final_data[i,n,:,:] = data_member[10,t,:,:]
        n = n + 1
        final_data[i,n,:,:] = data_member[11,t,:,:]
        n = n + 1

print(np.shape(final_data))

plt.contourf(final_data[0,0,:,:])
plt.savefig("/barnes-scratch/mafern/ForceSMIP/ForceSMIP/evaluate/figures/tempfig.png")

###############################
###############################
###############################
for i, member in enumerate(memberID):
    saveFilename = variable + "_" + memberID[i] + "_" + Tier + "_AutoEncoder_TeamPlanetGlitter_TEST"

    if PlotExample:
        if IS_Zonal_avg:
            plt.figure(dpi = (200), figsize = (6,4))
            cs = plt.pcolormesh(plev, lat, Ptrain[0,:,:], cmap = "Reds")
            cbar = plt.colorbar(cs,shrink=0.7,orientation='horizontal',label='Surface Air Temperature (C)', format='%.1f')#, pad=5)
            plt.title("Sample " + str(sample_n))
            plt.savefig(saveDirectoryPath + saveFilename + "sample")
        else:
            plt.figure(dpi = (200), figsize = (6,4))
            ax=plt.axes(projection= ccrs.PlateCarree())
            data, lonsr = add_cyclic_point(Ptrain[sample_n, :, :], coord=lon)
            cs = plt.pcolormesh(lonsr, lat, data, cmap = "Reds")
            ax.coastlines()
            cbar = plt.colorbar(cs,shrink=0.7,orientation='horizontal',label='Surface Air Temperature (C)', format='%.1f')#, pad=5)
            # cbar.set_label(colorbar_title)
            plt.title("Sample " + str(sample_n))
            plt.savefig(saveDirectoryPath + saveFilename + "sample")

    if CreateSaveFile:
        if IS_Zonal_avg:
            exists = os.path.isfile(saveDirectoryPath + saveFilename + ".nc")
            if exists:
                os.remove(saveDirectoryPath + saveFilename + ".nc")

            ts = nc.Dataset(saveDirectoryPath + saveFilename + ".nc", 'w' , format='NETCDF4')
            ts_plev = ts.createDimension('plev',plev_n)
            ts_lat = ts.createDimension('lat',lat_n)
            ts_time = ts.createDimension('time', np.size(Ptrain[:,0,0]))

            forced_component = ts.createVariable('forced','f4',('time', 'plev', 'lat'))
            forced_component[:,:,:] = final_data[i,:,:,:]

            ts.close()
        else:
            exists = os.path.isfile(saveDirectoryPath + saveFilename + ".nc")
            if exists:
                os.remove(saveDirectoryPath + saveFilename + ".nc")

            ts = nc.Dataset(saveDirectoryPath + saveFilename + ".nc", 'w' , format='NETCDF4')
            ts_lat = ts.createDimension('lat',lat_n)
            ts_lon = ts.createDimension('lon',lon_n)
            ts_time = ts.createDimension('time', np.size(Ptrain[:,0,0]))

            forced_component = ts.createVariable('forced','f4',('time','lat','lon'))
            forced_component[:,:,:] = final_data[i,:,:,:]

            ts.close()

    if CheckSavedFile:
    if IS_Zonal_avg:
        f = nc.Dataset("/barnes-scratch/mafern/ForceSMIP/ForceSMIP/SubmissionFiles/TOS_T1_AutoEncoder_TeamPlanetGlitter_TEST.nc")
        plotdata = f["forced"]
        
        plt.figure(dpi = (200), figsize = (6,4))
        cs = plt.pcolormesh(plev, lat, plotdata[0,:,:], cmap = "Reds")
        cbar = plt.colorbar(cs,shrink=0.7,orientation='horizontal',label='Surface Air Temperature (C)', format='%.1f')#, pad=5)
        plt.title("Sample " + str(sample_n))
        plt.savefig(saveDirectoryPath + saveFilename + "sampleOpen")
    else:
        f = nc.Dataset("/barnes-scratch/mafern/ForceSMIP/ForceSMIP/SubmissionFiles/TOS_T1_AutoEncoder_TeamPlanetGlitter_TEST.nc")
        plotdata = f["forced"]
        
        plt.figure(dpi = (200), figsize = (6,4))
        ax=plt.axes(projection= ccrs.PlateCarree())
        data, lonsr = add_cyclic_point(plotdata[sample_n, :, :], coord=lon)
        cs = plt.pcolormesh(lonsr, lat, data, cmap = "Reds")
        ax.coastlines()
        cbar = plt.colorbar(cs,shrink=0.7,orientation='horizontal',label='Surface Air Temperature (C)', format='%.1f')#, pad=5)
        # cbar.set_label(colorbar_title)
        plt.title("Sample " + str(sample_n))
        plt.savefig(saveDirectoryPath + saveFilename + "sampleOpen")