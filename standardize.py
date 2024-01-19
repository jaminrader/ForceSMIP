import numpy as np

class DataDoer():
    def __init__(self, Atrain_us, Aval_us, Atest_us, Ftrain_us, Fval_us, Itrain_us, Ival_us, settings,):
        self.Atrain_us = Atrain_us
        self.Aval_us = Aval_us
        self.Ftrain_us = Ftrain_us
        self.Fval_us = Fval_us
        self.Itrain_us = Itrain_us
        self.Ival_us = Ival_us
        self.Atest_us = Atest_us
        self.settings = settings
        
    def standardize(self):
        ### Standardize 
        
        if self.settings["output_std_method"] == "overall":
            # standardize the F maps by the training set, across all the data   
            axis = (0,1,2,)
        elif self.settings["output_std_method"] == "feature":
            # standardize the F maps by the training set, gridpoint-by-gridpoint     
            axis = (0,)
            
        self.fmean = np.nanmean(self.Ftrain_us, axis=axis)
        self.fstd  = np.nanstd(self.Ftrain_us, axis=axis)
        self.imean = np.nanmean(self.Itrain_us, axis=axis)
        self.istd  = np.nanstd(self.Itrain_us, axis=axis)
        
        # Add missing dimensions
        for i in range(self.Ftrain_us.ndim - self.fmean.ndim - 1):
            self.fmean = self.fmean[..., None]
            self.fstd = self.fstd[..., None]
            self.imean = self.imean[..., None]
            self.istd = self.istd[..., None]
        
        self.Ftrain = (self.Ftrain_us - self.fmean)/self.fstd
        self.Fval = (self.Fval_us - self.fmean)/self.fstd
        self.Itrain = (self.Itrain_us - self.imean)/self.istd
        self.Ival = (self.Ival_us - self.imean)/self.istd
        
        #standardize the A maps by themselves (except last dim, because that's the variables dimension
        if self.settings["input_std_method"] == "self":
            Aaxis = (1,2)
            self.Atrain = (self.Atrain_us - np.nanmean(self.Atrain_us, axis=Aaxis)[:, None, None, :]) / np.nanstd(self.Atrain_us, axis=Aaxis)[:, None, None, :]
            self.Aval = (self.Aval_us - np.nanmean(self.Aval_us, axis=Aaxis)[:, None, None, :]) / np.nanstd(self.Aval_us, axis=Aaxis)[:, None, None, :]
            self.Atest = (self.Atest_us - np.nanmean(self.Atest_us, axis=Aaxis)[:, None, None, :]) / np.nanstd(self.Atest_us, axis=Aaxis)[:, None, None, :]
        else:
        
            if self.settings["input_std_method"] == "overall":
                # standardize the F maps by the training set, across all the data   
                axis = (0,1,2,)
            elif self.settings["input_std_method"] == "feature":
                # standardize the F maps by the training set, gridpoint-by-gridpoint     
                axis = (0,)

            self.amean = np.nanmean(self.Atrain_us, axis=axis)
            self.astd = np.nanstd(self.Atrain_us, axis=axis)

            self.Atrain = (self.Atrain_us - self.amean)/self.astd
            self.Aval = (self.Aval_us - self.amean)/self.astd
            self.Atest = (self.Atest_us - self.amean)/self.astd
        
    def select(self):
        self.Xtrain = self.Atrain
        self.Xval = self.Aval
        self.Xtest = self.Atest
        if self.settings["target_component"] == "forced":
            self.Ttrain = self.Ftrain
            self.Tval = self.Fval
            self.tmean = self.fmean
            self.tstd = self.fstd
        elif self.settings["target_component"] == "internal":
            self.Ttrain = self.Itrain
            self.Tval = self.Ival
            self.tmean = self.imean
            self.tstd = self.istd
            
    def remove_nans(self):
        self.Xtrain = np.nan_to_num(self.Xtrain)
        self.Xtest = np.nan_to_num(self.Xtest)
        self.Xval = np.nan_to_num(self.Xval)
        self.Ttrain = np.nan_to_num(self.Ttrain)
        self.Tval = np.nan_to_num(self.Tval)
            
    def unstandardize(self, Ptrain, Pval, Ptest):
        print('Returning (Ptrain_us, Pval_us, Ptest_us)')
        Ptrain_us = Ptrain * self.tstd + self.tmean
        Pval_us = Pval * self.tstd + self.tmean
        Ptest_us = Ptest * self.tstd + self.tmean
        return Ptrain_us, Pval_us, Ptest_us
            
    def do(self):
        print('Returning (Xtrain, Xval, Xtest, Ttrain, Tval)')
        self.standardize()
        self.select()
        self.remove_nans()
        return self.Xtrain, self.Xval, self.Xtest, self.Ttrain, self.Tval
        