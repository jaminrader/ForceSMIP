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

def load_model(model,var,timecut="Tier1",ntrainmems=20):
    print(root_dir, cmipTable[var], var, model)

    filelist = root_dir + "Training/" + cmipTable[var] + "/" + var + "/" + model + "/" + var + "*.nc"
    filelist = glob.glob(filelist)
    filelist = sorted(filelist)

    timebds = evalPeriods[timecut]

    all_ens = []
    for ifile,file in enumerate(filelist[:ntrainmems]):
        ds = xr.open_dataset(file)
        varin = ds[cmipVar[var]]
        varcut = varin.sel(time=slice(timebds[0],timebds[1]))
        varcut = varcut.groupby('time.year').mean()  

        varcut=varcut.assign_coords({"variant":ifile+1})
        all_ens.append(varcut)

    all_ens = xr.concat(all_ens,dim="variant")
    
    return all_ens

def make_X_data(models = ["CESM2","MIROC6","CanESM5"], var = "tos", timecut = "Tier1", nmems = 20):
    nmodels = len(models)

    timebds = evalPeriods[timecut]
    time1 = int(timebds[0][:4])
    time2 = int(timebds[1][:4])

    ntime = time2-time1+1

    X = np.empty((ntime*nmems*nmodels,nlat,nlon))+np.nan

    for imod, model in enumerate(models):

        da=load_model(model,var,timecut,nmems)
        da_np = np.asarray(da)
        Xloop = np.empty((nmems*ntime,nlat,nlon))+np.nan

        for imem in range(nmems):
            Xloop[imem*ntime:(imem+1)*ntime,:,:] = da_np[imem,:,:,:]

        X[nmems*ntime*imod:nmems*ntime*(imod+1),:,:] = Xloop

    Xbad = np.mean(X,axis=0)
    X[:,np.isnan(Xbad)] = 0
    
    return X


def make_data(models = ["CESM2","MIROC6","CanESM5"], var = "tos", timecut = "Tier1", mems = np.arange(20)):
    
    # models: a list of models from     "MIROC6
    #                                    "CESM2"
    #                                   "CanESM5"
    #                                   "MPI-ESM1-2-LR"
    #                                   "MIROC-ES2L"
    
    # var: a variable string from     "pr"
    #                                 "psl"
    #                                 "tas"
    #                                 "zmta"
    #                                 "tos"
    #                                 "siconc"
    #                                 "monmaxpr":
    #                                 "monmaxtasmax"
    #                                 "monmintasmin"
    
    # timecut: from the evaluation Tiers 
    #     "Tier1" ("1950-01-01", "2022-12-31"),
    #     "Tier2" ("1900-01-01", "2022-12-31"),
    #     "Tier3" ("1979-01-01", "2022-12-31"),
    
    # mems: a list of members to use
    
    nmodels = len(models)

    timebds = evalPeriods[timecut]
    time1 = int(timebds[0][:4])
    time2 = int(timebds[1][:4])

    ntime = time2-time1+1

    nmems = len(mems)
    
    Yforced = np.empty((ntime*nmems*nmodels,nlat,nlon))+np.nan
    Yinternal = np.empty((ntime*nmems*nmodels,nlat,nlon))+np.nan
    Xfull = np.empty((ntime*nmems*nmodels,nlat,nlon))+np.nan
    
    for imod, model in enumerate(models):

        nfullmems = fullmems[model]
        
        da=load_model(model,var,timecut,nfullmems)
        
        da_np = np.asarray(da)
        da_f_np = np.mean(da_np,axis=0,keepdims=True)
        da_i_np = da_np-da_f_np
        
        Yloop = np.empty((nmems*ntime,nlat,nlon))+np.nan
        Yloop_i = np.empty((nmems*ntime,nlat,nlon))+np.nan
        Yloop_f = np.empty((nmems*ntime,nlat,nlon))+np.nan
        
        for imem, mem in enumerate(mems):
            Yloop_f[imem*ntime:(imem+1)*ntime] = da_f_np
            Yloop_i[imem*ntime:(imem+1)*ntime] = da_i_np[mem,:,:,:]
            Yloop[imem*ntime:(imem+1)*ntime] = da_np[mem,:,:,:]

        Xfull[nmems*ntime*imod:nmems*ntime*(imod+1)] = Yloop
        Yforced[nmems*ntime*imod:nmems*ntime*(imod+1)] = Yloop_f
        Yinternal[nmems*ntime*imod:nmems*ntime*(imod+1)] = Yloop_i
    
    return Xfull,Yforced,Yinternal

def save_npz(settings, A_train, F_train, I_train, A_val, F_val, I_val):
    os.system('mkdir ' +  settings['npz_dir'])
    np.savez(settings['npz_dir'] + settings['exp_name'] + '.npz', At = A_train, Ft = F_train, It = I_train, 
             Av = A_val, Fv = F_val, Iv = I_val)

def load_npz(settings):
    npzdat = np.load(settings['npz_dir'] + settings['exp_name'] + '.npz')
    At, Ft, It, Av, Fv, Iv = npzdat['At'], npzdat['Ft'], npzdat['It'], npzdat['Av'], npzdat['Fv'], npzdat['Iv']
    return At, Ft, It, Av, Fv, Iv
    

def stack_variable(X_tuple):
    
    Xout = np.stack(X_tuple,axis=-1)
    
    return Xout
    
def make_eval_mem(evalmem="1H",var="tos",timecut="Tier1"):
    
    filelist = root_dir + "Evaluation-"+timecut+"/" + cmipTable[var] + "/" + var + "/" + var + "*" + evalmem + "*.nc"
    file = glob.glob(filelist)[0]
    ds = xr.open_dataset(file)
    varin = ds[cmipVar[var]]
    varmean = varin.groupby("time.year").mean()
    varmean_np = np.asarray(varmean)
    
    return varmean_np

def make_eval_mem_monthly(evalmem="1H",var="tos",timecut="Tier1"):
    
    filelist = root_dir + "Evaluation-"+timecut+"/" + cmipTable[var] + "/" + var + "/" + var + "*" + evalmem + "*.nc"
    file = glob.glob(filelist)[0]
    ds = xr.open_dataset(file)
    varin = ds[cmipVar[var]]
    #varmean = varin.groupby("time.year").mean()
    varin_np = np.asarray(varin)
    
    return varin_np