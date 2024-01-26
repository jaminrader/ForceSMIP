import numpy as np

def add_data_to_experiment(settings, data_name):
    for key in data_dictionary[data_name]:
        settings[key] = data_dictionary[data_name][key]
    return settings

def get_experiment(exp_name, settings_overwrite=None):

    if settings_overwrite is None:
        settings = experiments[exp_name]
    else:
        settings = settings_overwrite[exp_name]
    settings["exp_name"] = exp_name
    settings["npz_dir"] = 'exp_data/'
    settings["pred_dir"] = 'saved_predictions/'
    settings = add_data_to_experiment(settings)

    return settings

data_dictionary = {
    # "data_template":{
    #     "input_variable": ["tos",], # List of variables for input
    #     "target_variable": "tos", # Single variable output
    #     "train_models"  : ["MIROC6","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L",], # Models for training
    #     "val_models"    : ["MIROC6","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L",], # Models for validations
    #     "test_models"   : ["CESM2",], # Models for testing
    #     "train_members" : np.arange(18), # Ensembles members for training
    #     "val_members"   : np.arange(18, 25), # Ensemble members for validation
    #     "test_members"  : np.arange(25), # Ensemble members for testing
    #     "time_range" : "Tier1", # ForceSMIP Tier
    # },
    "Train4_Val4_CESM2_tos_tos":{
        "input_variable": ["tos",],
        "target_variable": "tos",
        "train_models"  : ["MIROC6","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L",],
        "val_models"    : ["MIROC6","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L",],
        "test_models"   : ["CESM2",],
        "train_members" : np.arange(18),
        "val_members"   : np.arange(18, 25),
        "test_members"  : np.arange(25),
        "time_range" : "Tier1",
    },

    "Train4_Val4_MIROC6_tos_tos":{
        "input_variable": ["tos",],
        "target_variable": "tos",
        "train_models"  : ["CESM2","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L",],
        "val_models"    : ["CESM2","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L",],
        "test_models"   : ["MIROC6",],
        "train_members" : np.arange(18),
        "val_members"   : np.arange(18, 25),
        "test_members"  : np.arange(25),
        "time_range" : "Tier1",
    },

    "Train4_Val4_CanESM5_tos_tos":{
        "input_variable": ["tos",],
        "target_variable": "tos",
        "train_models"  : ["CESM2","MIROC6","MPI-ESM1-2-LR","MIROC-ES2L",],
        "val_models"    : ["CESM2","MIROC6","MPI-ESM1-2-LR","MIROC-ES2L",],
        "test_models"   : ["CanESM5",],
        "train_members" : np.arange(18),
        "val_members"   : np.arange(18, 25),
        "test_members"  : np.arange(25),
        "time_range" : "Tier1",
    },

    "Train4_Val4_MPI-ESM1-2-LR_tos_tos":{
        "input_variable": ["tos",],
        "target_variable": "tos",
        "train_models"  : ["CESM2","MIROC6","CanESM5","MIROC-ES2L",],
        "val_models"    : ["CESM2","MIROC6","CanESM5","MIROC-ES2L",],
        "test_models"   : ["MPI-ESM1-2-LR",],
        "train_members" : np.arange(18),
        "val_members"   : np.arange(18, 25),
        "test_members"  : np.arange(25),
        "time_range" : "Tier1",
    },

    "Train4_Val4_MIROC-ES2L_tos_tos":{
        "input_variable": ["tos",],
        "target_variable": "tos",
        "train_models"  : ["CESM2","MIROC6","CanESM5","MPI-ESM1-2-LR",],
        "val_models"    : ["CESM2","MIROC6","CanESM5","MPI-ESM1-2-LR",],
        "test_models"   : ["MIROC-ES2L",],
        "train_members" : np.arange(18),
        "val_members"   : np.arange(18, 25),
        "test_members"  : np.arange(25),
        "time_range" : "Tier1",
    },
}

experiments = {
    # "exp_template" : {
    #     #Ran by: 
    #     #Metrics:
    #     #Notes:
    #     "seed"          : 0 ,
    #     "learn_rate"    : .001,
    #     "patience"      : 50,
    #     "batch_size"    : 64,
    #     "max_epochs"    : 5_000,
    #     "metric"        : "",
        
    #     "input_std_method"    : "", #self, feature, overall
    #     "output_std_method"   : "", # feature, overall
    #     "target_component": "internal", #forced or internal

    #     "CNN_blocks"    : [2],
    #     "CNN_filters"   : [32],
    #     "CNN_kernals"   : [3],
    #     "CNN_strides"   : [1],

    #     "encoding_nodes" : [20, 20, 10],
    #     "code_nodes"     : 1,
    #     "activation"     : "linear",
    #     "variational_loss": .001
    # },

    "internal_feature" : {

        # Standardization + prediction component
        "target_component": "internal",
        "input_std_method"    : "feature",
        "output_std_method"   : "feature",

        # Network specs
        "seed"          : 0 ,
        "learn_rate" : 0.001,
        "patience"      : 100,
        "batch_size"    : 64,
        "max_epochs"    : 2000,
        
        # Architecture specs
        "encoding_nodes" : [10,10,],
        "code_nodes"     : 3,
        "activation"     : "tanh",
        "variational_loss": 0.0001
    },

    "forced_feature" : {

        # Standardization + prediction component
        "target_component": "forced",
        "input_std_method"    : "feature",
        "output_std_method"   : "feature",

        # Network specs
        "seed"          : 0 ,
        "learn_rate" : 0.001,
        "patience"      : 100,
        "batch_size"    : 64,
        "max_epochs"    : 2000,
        
        # Architecture specs
        "encoding_nodes" : [10,10,],
        "code_nodes"     : 3,
        "activation"     : "tanh",
        "variational_loss": 0.0001
    },
}

### Add data_name to the experiments

for exp_name in experiments.copy():
    for data_name in data_dictionary:
        experiments[exp_name + '_' + data_name] = experiments[exp_name].copy()
        experiments[exp_name + '_' + data_name]["data_name"] = data_name
        add_data_to_experiment(experiments[exp_name + '_' + data_name], data_name)
print(experiments)