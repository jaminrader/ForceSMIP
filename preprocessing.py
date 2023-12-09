machine = 'casper' # 'casper' or 'asha'

# I/O / data wrangling
import glob
import numpy as np
import xarray as xr
import gc
import os

if machine == 'casper':
    root_dir = "/glade/campaign/cgd/cas/asphilli/ForceSMIP/"
elif machine == 'asha':
    root_dir = "/barnes-scratch/DATA/ForceSMIP/"
########################################################
# remember to remove this
root_dir = '/Volumes/Data/Martin_Fernandez/ForceSMIP/'
    
cmipTable = {
    "pr": "Amon",
    "psl": "Amon",
    "tas": "Amon",
    "zmta": "Amon",
    "tos": "Omon",
    "siconc": "OImon",
    "monmaxpr": "Aday",
    "monmaxtasmax": "Aday",
    "monmintasmin": "Aday",
}
cmipVar = {
    "pr": "pr",
    "psl": "psl",
    "tas": "tas",
    "zmta": "ta",
    "tos": "tos",
    "siconc": "siconc",
    "monmaxpr": "pr",
    "monmaxtasmax": "tasmax",
    "monmintasmin": "tasmin",
}
evalPeriods = {
    "Tier1": ("1950-01-01", "2022-12-31"),
    "Tier2": ("1900-01-01", "2022-12-31"),
    "Tier3": ("1979-01-01", "2022-12-31"),
}
fullmems = {
    "MIROC6": 50,
    "CESM2": 50,
    "CanESM5": 25,
    "MPI-ESM1-2-LR": 30,
    "MIROC-ES2L": 30,
}
nlat = 72
nlon = 144


def load_model(model, var, timecut="Tier1", ntrainmems=20):
    print(root_dir, cmipTable[var], var, model)
    # get the list of all files (members) for this model and variable
    filelist = root_dir + "Training/" + cmipTable[var] + "/" + var + "/" + model + "/" + var + "*.nc"
    filelist = glob.glob(filelist)
    filelist = sorted(filelist)
    # time period to cut on
    timebds = evalPeriods[timecut]

    # go through each file and load, append
    all_ens = []
    for ifile, file in enumerate(filelist[:ntrainmems]):
        ds = xr.open_dataset(file)
        # select the variable
        varin = ds[cmipVar[var]]
        # get the correct time period
        varcut = varin.sel(time=slice(timebds[0],timebds[1]))
        # make data yearly
        #########################################################################
        # currently always using yearly data
        #########################################################################
        varcut = varcut.groupby('time.year').mean()  
        # add coordinate for concatenating after the loop
        varcut = varcut.assign_coords({"variant":ifile+1})
        # append into a list
        all_ens.append(varcut)
    # get all members into a single xarray data array
    all_ens = xr.concat(all_ens, dim="variant")
    
    return all_ens


def make_data(models=["CESM2", "MIROC6", "CanESM5"], var="tos", timecut="Tier1", mems=np.arange(20)):
    """ 
    models: a list of models from     "MIROC6
                                       "CESM2"
                                      "CanESM5"
                                      "MPI-ESM1-2-LR"
                                      "MIROC-ES2L"
    
    var: a variable string from     "pr"
                                    "psl"
                                    "tas"
                                    "zmta"
                                    "tos"
                                    "siconc"
                                    "monmaxpr":
                                    "monmaxtasmax"
                                    "monmintasmin"
    
    timecut: from the evaluation Tiers 
        "Tier1" ("1950-01-01", "2022-12-31"),
        "Tier2" ("1900-01-01", "2022-12-31"),
        "Tier3" ("1979-01-01", "2022-12-31"),
    
    mems: a list of members to use
    """
    # number of members requested
    nmems = len(mems)

    # go through each model separately (targets are model specific)
    for imod, model in enumerate(models):
        # get the number of members available for this model
        nfullmems = fullmems[model]
        # check that we can load the number of members requested
        assert nmems <= nfullmems, f"more members requested ({nmems}) than available ({nfullmems})"
        # load the members for this model/variable into an xarray
        # returns the correct time period given the tier
        # note it loads all members, so the forced response is correct, then picks members based on mems
        da = load_model(model, var, timecut, nfullmems)
        # convert from xarray to a numpy array: dimensions are [members, time, lat, lon]
        da_np = np.asarray(da)
        # get the forced response as the ensemble mean across all members (0th dimension)
        da_f_np = np.mean(da_np, axis=0, keepdims=True)
        # internal variability is the difference between the full signal and the forced response
        # only use the requested members
        da_i_np = da_np[mems] - da_f_np
        # reshape these to dimensions: [samples, lat, lon]
        da_np = np.reshape(da_np[mems], (-1, nlat, nlon))
        # for forced response, repeat nmems times to match full and internal
        da_f_np = np.reshape([da_f_np]*nmems, (-1, nlat, nlon))
        da_i_np = np.reshape(da_i_np, (-1, nlat, nlon))
        # stack the models together        
        if imod == 0:
            Xfull = da_np
            Yforced = da_f_np
            Yinternal = da_i_np
        else:
            Xfull = np.vstack([Xfull, da_np])
            Yforced = np.vstack([Yforced, da_f_np])
            Yinternal = np.vstack([Yinternal, da_i_np])
    
    return Xfull, Yforced, Yinternal


def save_npz(settings, A_train, F_train, I_train, A_val, F_val, I_val):
    os.system('mkdir ' +  settings['npz_dir'])
    np.savez(settings['npz_dir'] + settings['exp_name'] + '.npz', At=A_train, Ft=F_train, It=I_train, 
             Av=A_val, Fv=F_val, Iv=I_val)


def load_npz(settings):
    npzdat = np.load(settings['npz_dir'] + settings['exp_name'] + '.npz')
    At, Ft, It, Av, Fv, Iv = npzdat['At'], npzdat['Ft'], npzdat['It'], npzdat['Av'], npzdat['Fv'], npzdat['Iv']
    return At, Ft, It, Av, Fv, Iv
    

def stack_variable(X_tuple):
    Xout = np.stack(X_tuple, axis=-1)
    return Xout
    

def make_eval_mem(evalmem="1H", var="tos", timecut="Tier1"):
    "evalmem can be 1A through 1J, so determines the member"
    # get a member from one of the evaulation data sets
    filelist = root_dir + "Evaluation-"+timecut+"/" + cmipTable[var] + "/" + var + "/" + var + "*" + evalmem + "*.nc"
    # glob the full file name (zero index to remove from list)
    file = glob.glob(filelist)[0]
    ds = xr.open_dataset(file)
    # get the specified variable and make yearly
    varin = ds[cmipVar[var]]
    varmean = varin.groupby("time.year").mean()
    # return as a numpy array    
    return np.asarray(varmean)


def make_eval_mem_monthly(evalmem="1H", var="tos", timecut="Tier1"):
    "evalmem can be 1A through 1J, so determines the member"
    # get a member from one of the evaulation data sets
    filelist = root_dir + "Evaluation-"+timecut+"/" + cmipTable[var] + "/" + var + "/" + var + "*" + evalmem + "*.nc"
    # glob the full file name (zero index to remove from list)
    file = glob.glob(filelist)[0]
    # get the specified variable
    ds = xr.open_dataset(file)
    varin = ds[cmipVar[var]]
    # return as a numpy array 
    return np.asarray(varin)


# def make_X_data(models=["CESM2","MIROC6","CanESM5"], var="tos", timecut="Tier1", nmems=20):
#     nmodels = len(models)

#     timebds = evalPeriods[timecut]
#     time1 = int(timebds[0][:4])
#     time2 = int(timebds[1][:4])

#     ntime = time2 - time1 + 1

#     X = np.empty((ntime*nmems*nmodels, nlat, nlon)) + np.nan

#     for imod, model in enumerate(models):

#         da = load_model(model, var, timecut, nmems)
#         da_np = np.asarray(da)
#         Xloop = np.empty((nmems*ntime, nlat, nlon)) + np.nan

#         for imem in range(nmems):
#             Xloop[imem*ntime:(imem+1)*ntime, :, :] = da_np[imem, :, :, :]

#         X[nmems*ntime*imod:nmems*ntime*(imod+1), :, :] = Xloop

#     Xbad = np.mean(X, axis=0)
#     X[:, np.isnan(Xbad)] = 0
    
#     return X
