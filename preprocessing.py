machine = 'casper' # 'casper' or 'asha'

# I/O / data wrangling
import glob
import numpy as np
import xarray as xr
import gc

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
nlat = 72
nlon = 144

def load_model(model,var,timecut="Tier1",ntrainmems=20):

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


def make_Y_data(models = ["CESM2","MIROC6","CanESM5"], var = "tos", timecut = "Tier1", nmems = 20):
    
    nmodels = len(models)

    timebds = evalPeriods[timecut]
    time1 = int(timebds[0][:4])
    time2 = int(timebds[1][:4])

    ntime = time2-time1+1

    Yforced = np.empty((ntime*nmems*nmodels,nlat,nlon))+np.nan
    Yinternal = np.empty((ntime*nmems*nmodels,nlat,nlon))+np.nan
    Yfull = np.empty((ntime*nmems*nmodels,nlat,nlon))+np.nan
    
    for imod, model in enumerate(models):

        nfullmems = fullmems[model]
        
        da=load_model(model,var,timecut,nfullmems)
        
        da_np = np.asarray(da)
        da_f_np = np.mean(da_np,axis=0,keepdims=True)
        da_i_np = da_np-da_f_np
        
        Yloop = np.empty((nmems*ntime,nlat,nlon))+np.nan
        Yloop_i = np.empty((nmems*ntime,nlat,nlon))+np.nan
        Yloop_f = np.empty((nmems*ntime,nlat,nlon))+np.nan
        
        for imem in range(nmems):
            Yloop_f[imem*ntime:(imem+1)*ntime] = da_f_np
            Yloop_i[imem*ntime:(imem+1)*ntime] = da_i_np[imem,:,:,:]
            Yloop[imem*ntime:(imem+1)*ntime] = da_np[imem,:,:,:]

        Yfull[nmems*ntime*imod:nmems*ntime*(imod+1)] = Yloop
        Yforced[nmems*ntime*imod:nmems*ntime*(imod+1)] = Yloop_f
        Yinternal[nmems*ntime*imod:nmems*ntime*(imod+1)] = Yloop_i
    
    return Yforced,Yinternal,Yfull

    