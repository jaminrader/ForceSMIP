import numpy as np

def get_experiment(exp_name, base_exp_name=None, settings_overwrite=None):

    if settings_overwrite is None:
        settings = get_experiment_settings(exp_name)
    else:
        settings = settings_overwrite[exp_name]
    settings["exp_name"] = exp_name
    settings["npz_dir"] = 'exp_data/'
    settings["pred_dir"] = 'saved_predictions/'

    # if base_exp_name is not None:
        # settings["base_exp_name"] = base_exp_name

    # if "ignore_smooth_warning" in list(settings.keys()) and settings["ignore_smooth_warning"] == True:
        # print('IGNORING SMOOTHING WARNINGS')
    # else:
        # assert settings["lead_time"] >= 0, f"lead_time must be non-negative."
        # assert settings["smooth_len_input"] >= 0, f"input smoothing must be non-negative."
        # assert settings["smooth_len_output"] <= 0, f"output smoothing must be non-positive."

    return settings

def get_experiment_settings(exp_name):

    return experiments[exp_name]

experiments = {
    "exp_template" : {
        #Ran by: 
        #Metrics:
        #Notes:
        "seed"          : 0 ,
        "learn_rate"    : .001,
        "patience"      : 50,
        "batch_size"    : 64,
        "max_epochs"    : 5_000,
        "metric"        : "",
        
        "input_std_method"    : "", #self, feature, overall
        "output_std_method"   : "", # feature, overall
        "input_variable": ["tos","pr",],
        "target_variable": "tos",
        "target_component": "internal", #forced or internal
        "train_models"  : [],
        "val_models"    : [],
        "test_models"   : [],
        "train_members" : [],
        "val_members"   : [],
        "test_members"  : [],
        "time_range"    : 0,

        "CNN_blocks"    : [2],
        "CNN_filters"   : [32],
        "CNN_kernals"   : [3],
        "CNN_strides"   : [1],

        "encoding_nodes" : [20, 20, 10],
        "code_nodes"     : 1,
        "activation"     : "linear",
        "variational_loss": .001
    },
    
    "test_exp" : {
        "seed"          : 0 ,
        "learn_rate" : 0.001,
        "patience"      : 50,
        "batch_size"    : 64,
        "max_epochs"    : 10,
        "metric"        : "",
        
        "input_std_method"    : "self", #self, feature, overall
        "output_std_method"   : "feature", # feature, overall
        "input_variable": ["tos","pr",],
        "target_variable": "tos",
        "target_component": "internal", #forced or internal
        "train_models"  : ["CESM2",],
        "val_models"    : ["CESM2",],
        "test_models"   : None,
        "train_members" : np.arange(10),
        "val_members"   : np.arange(11, 13),
        "test_members"  : None,
        "time_range"    : "Tier1",

        "CNN_blocks"    : [2],
        "CNN_filters"   : [32],
        "CNN_kernals"   : [3],
        "CNN_strides"   : [1],

        "encoding_nodes" : [20, 20, 10],
        "code_nodes"     : 10,
        "activation"     : "linear",
        "variational_loss": 0.0001
    },
    
    "test_exp_tos" : {
        "seed"          : 0 ,
        "learn_rate" : 0.001,
        "patience"      : 50,
        "batch_size"    : 64,
        "max_epochs"    : 10,
        "metric"        : "",
        
        "input_std_method"    : "self", #self, feature, overall
        "output_std_method"   : "feature", # feature, overall
        "input_variable": ["tos",],
        "target_variable": "tos",
        "target_component": "internal", #forced or internal
        "train_models"  : ["CESM2",],
        "val_models"    : ["CESM2",],
        "test_models"   : None,
        "train_members" : np.arange(10),
        "val_members"   : np.arange(11, 13),
        "test_members"  : None,
        "time_range"    : "Tier1",

        "CNN_blocks"    : [2],
        "CNN_filters"   : [32],
        "CNN_kernals"   : [3],
        "CNN_strides"   : [1],

        "encoding_nodes" : [20, 20, 10],
        "code_nodes"     : 10,
        "activation"     : "linear",
        "variational_loss": 0.0001
    },
    
    "test_tos" : {
        "seed"          : 0 ,
        "learn_rate" : 0.001,
        "patience"      : 50,
        "batch_size"    : 64,
        "max_epochs"    : 200,
        "metric"        : "",
        
        "input_std_method"    : "self", #self, feature, overall
        "output_std_method"   : "feature", # feature, overall
        "input_variable": ["tos",],
        "target_variable": "tos",
        "target_component": "internal", #forced or internal
        "train_models"  : ["CESM2",],
        "val_models"    : ["CESM2",],
        "test_models"   : None,
        "train_members" : np.arange(10),
        "val_members"   : np.arange(11, 13),
        "test_members"  : None,
        "time_range"    : "Tier1",

        "CNN_blocks"    : [2],
        "CNN_filters"   : [32],
        "CNN_kernals"   : [3],
        "CNN_strides"   : [1],

        "encoding_nodes" : [20, 20, 10],
        "code_nodes"     : 100,
        "activation"     : "linear",
        "variational_loss": 0.0001
    },
    
    "tos_exp_0" : {
        "seed"          : 0 ,
        "learn_rate" : 0.001,
        "patience"      : 50,
        "batch_size"    : 64,
        "max_epochs"    : 200,
        "metric"        : "",
        
        "input_std_method"    : "self", #self, feature, overall
        "output_std_method"   : "feature", # feature, overall
        "input_variable": ["tos",],
        "target_variable": "tos",
        "target_component": "internal", #forced or internal
        "train_models"  : ["CESM2",],
        "val_models"    : ["CESM2",],
        "test_models"   : None,
        "train_members" : np.arange(10),
        "val_members"   : np.arange(11, 13),
        "test_members"  : None,
        "time_range"    : "Tier1",

        "CNN_blocks"    : [2],
        "CNN_filters"   : [32],
        "CNN_kernals"   : [3],
        "CNN_strides"   : [1],

        "encoding_nodes" : [100,100,],
        "code_nodes"     : 100,
        "activation"     : "tanh",
        "variational_loss": 0.0001
    },
    
    "tos_exp_1" : {
        "seed"          : 0 ,
        "learn_rate" : 0.001,
        "patience"      : 50,
        "batch_size"    : 64,
        "max_epochs"    : 200,
        "metric"        : "",
        
        "input_std_method"    : "self", #self, feature, overall
        "output_std_method"   : "feature", # feature, overall
        "input_variable": ["tos",],
        "target_variable": "tos",
        "target_component": "internal", #forced or internal
        "train_models"  : ["CESM2","MIROC6","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L",],
        "val_models"    : ["CESM2","MIROC6","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L",],
        "test_models"   : None,
        "train_members" : np.arange(10),
        "val_members"   : np.arange(11, 13),
        "test_members"  : None,
        "time_range"    : "Tier1",

        "CNN_blocks"    : [2],
        "CNN_filters"   : [32],
        "CNN_kernals"   : [3],
        "CNN_strides"   : [1],

        "encoding_nodes" : [100,100,],
        "code_nodes"     : 100,
        "activation"     : "tanh",
        "variational_loss": 0.0001
    },
    
    "tos_exp_forced0" : {
        "seed"          : 0 ,
        "learn_rate" : 0.001,
        "patience"      : 50,
        "batch_size"    : 64,
        "max_epochs"    : 200,
        "metric"        : "",
        
        "input_std_method"    : "feature", #self, feature, overall
        "output_std_method"   : "feature", # feature, overall
        "input_variable": ["tos",],
        "target_variable": "tos",
        "target_component": "forced", #forced or internal
        "train_models"  : ["CESM2","MIROC6","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L",],
        "val_models"    : ["CESM2","MIROC6","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L",],
        "test_models"   : None,
        "train_members" : np.arange(10),
        "val_members"   : np.arange(11, 13),
        "test_members"  : None,
        "time_range"    : "Tier1",

        "CNN_blocks"    : [2],
        "CNN_filters"   : [32],
        "CNN_kernals"   : [3],
        "CNN_strides"   : [1],

        "encoding_nodes" : [100,100,],
        "code_nodes"     : 100,
        "activation"     : "tanh",
        "variational_loss": 0.0001
    },
    
        "tos_exp_forced1" : {
        "seed"          : 0 ,
        "learn_rate" : 0.001,
        "patience"      : 50,
        "batch_size"    : 64,
        "max_epochs"    : 200,
        "metric"        : "",
        
        "input_std_method"    : "self", #self, feature, overall
        "output_std_method"   : "feature", # feature, overall
        "input_variable": ["tos",],
        "target_variable": "tos",
        "target_component": "forced", #forced or internal
        "train_models"  : ["CESM2","MIROC6","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L",],
        "val_models"    : ["CESM2","MIROC6","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L",],
        "test_models"   : None,
        "train_members" : np.arange(10),
        "val_members"   : np.arange(11, 13),
        "test_members"  : None,
        "time_range"    : "Tier1",

        "CNN_blocks"    : [2],
        "CNN_filters"   : [32],
        "CNN_kernals"   : [3],
        "CNN_strides"   : [1],

        "encoding_nodes" : [100,100,],
        "code_nodes"     : 100,
        "activation"     : "tanh",
        "variational_loss": 0.0001
    },
    
    "pr_exp_0" : {
        "seed"          : 0 ,
        "learn_rate" : 0.001,
        "patience"      : 50,
        "batch_size"    : 64,
        "max_epochs"    : 200,
        "metric"        : "",
        
        "input_std_method"    : "self", #self, feature, overall
        "output_std_method"   : "feature", # feature, overall
        "input_variable": ["pr",],
        "target_variable": "pr",
        "target_component": "internal", #forced or internal
        "train_models"  : ["CESM2",],
        "val_models"    : ["CESM2",],
        "test_models"   : None,
        "test_members"  : None,
        "train_members" : np.arange(20),
        "val_members"   : np.arange(21, 25),
        "time_range"    : "Tier1",

        "CNN_blocks"    : [2],
        "CNN_filters"   : [32],
        "CNN_kernals"   : [3],
        "CNN_strides"   : [1],

        "encoding_nodes" : [100,100,],
        "code_nodes"     : 100,
        "activation"     : "tanh",
        "variational_loss": 0.0001
    },
    
    "pr_exp_1" : {
        "seed"          : 0 ,
        "learn_rate" : 0.001,
        "patience"      : 50,
        "batch_size"    : 64,
        "max_epochs"    : 200,
        "metric"        : "",
        
        "input_std_method"    : "self", #self, feature, overall
        "output_std_method"   : "feature", # feature, overall
        "input_variable": ["pr",],
        "target_variable": "pr",
        "target_component": "internal", #forced or internal
        "train_models"  : ["CESM2","MIROC6","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L",],
        "val_models"    : ["CESM2","MIROC6","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L",],
        "test_models"   : None,
        "train_members" : np.arange(10),
        "val_members"   : np.arange(11, 13),
        "test_members"  : None,
        "time_range"    : "Tier1",

        "CNN_blocks"    : [2],
        "CNN_filters"   : [32],
        "CNN_kernals"   : [3],
        "CNN_strides"   : [1],

        "encoding_nodes" : [100,100,],
        "code_nodes"     : 100,
        "activation"     : "tanh",
        "variational_loss": 0.0001
    },


    "pr_exp_internal" : {
        "seed"          : 0 ,
        "learn_rate" : 0.001,
        "patience"      : 100,
        "batch_size"    : 64,
        "max_epochs"    : 2000,
        "metric"        : "",
        
        "input_std_method"    : "self", #self, feature, overall
        "output_std_method"   : "feature", # feature, overall
        "input_variable": ["pr", "tas", "psl", "tos"],
        "target_variable": "pr",
        "target_component": "internal", #forced or internal
        "train_models"  : ["CESM2", "MIROC6", "CanESM5", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "val_models"    : ["CESM2", "MIROC6", "CanESM5", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "test_models"   : ["CESM2", "MIROC6", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "train_members" : np.arange(22),
        "val_members"   : np.arange(22, 25),
        "test_members"  : np.arange(25, 30),
        "time_range"    : "Tier1",

        "CNN_blocks"    : [2],
        "CNN_filters"   : [32],
        "CNN_kernals"   : [3],
        "CNN_strides"   : [1],

        "encoding_nodes" : [100,100,],
        "code_nodes"     : 100,
        "activation"     : "tanh",
        "variational_loss": 0.0001
    },

    "pr_exp_forced" : {
        "seed"          : 0 ,
        "learn_rate" : 0.001,
        "patience"      : 100,
        "batch_size"    : 64,
        "max_epochs"    : 2000,
        "metric"        : "",
        
        "input_std_method"    : "self", #self, feature, overall
        "output_std_method"   : "feature", # feature, overall
        "input_variable": ["pr", "tas", "psl", "tos"],
        "target_variable": "pr",
        "target_component": "forced", #forced or internal
        "train_models"  : ["CESM2", "MIROC6", "CanESM5", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "val_models"    : ["CESM2", "MIROC6", "CanESM5", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "test_models"   : ["CESM2", "MIROC6", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "train_members" : np.arange(22),
        "val_members"   : np.arange(22, 25),
        "test_members"  : np.arange(25, 30),
        "time_range"    : "Tier1",

        "CNN_blocks"    : [2],
        "CNN_filters"   : [32],
        "CNN_kernals"   : [3],
        "CNN_strides"   : [1],

        "encoding_nodes" : [100,100,],
        "code_nodes"     : 100,
        "activation"     : "tanh",
        "variational_loss": 0.0001
    },

    "tos_exp_internal" : {
        "seed"          : 0 ,
        "learn_rate" : 0.001,
        "patience"      : 100,
        "batch_size"    : 64,
        "max_epochs"    : 2000,
        "metric"        : "",
        
        "input_std_method"    : "self", #self, feature, overall
        "output_std_method"   : "feature", # feature, overall
        "input_variable": ["pr", "tas", "psl", "tos"],
        "target_variable": "tos",
        "target_component": "internal", #forced or internal
        "train_models"  : ["CESM2", "MIROC6", "CanESM5", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "val_models"    : ["CESM2", "MIROC6", "CanESM5", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "test_models"   : ["CESM2", "MIROC6", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "train_members" : np.arange(22),
        "val_members"   : np.arange(22, 25),
        "test_members"  : np.arange(25, 30),
        "time_range"    : "Tier1",

        "CNN_blocks"    : [2],
        "CNN_filters"   : [32],
        "CNN_kernals"   : [3],
        "CNN_strides"   : [1],

        "encoding_nodes" : [100,100,],
        "code_nodes"     : 100,
        "activation"     : "tanh",
        "variational_loss": 0.0001
    },

    "tos_exp_forced" : {
        "seed"          : 0 ,
        "learn_rate" : 0.001,
        "patience"      : 100,
        "batch_size"    : 64,
        "max_epochs"    : 2000,
        "metric"        : "",
        
        "input_std_method"    : "self", #self, feature, overall
        "output_std_method"   : "feature", # feature, overall
        "input_variable": ["pr", "tas", "psl", "tos"],
        "target_variable": "tos",
        "target_component": "forced", #forced or internal
        "train_models"  : ["CESM2", "MIROC6", "CanESM5", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "val_models"    : ["CESM2", "MIROC6", "CanESM5", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "test_models"   : ["CESM2", "MIROC6", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "train_members" : np.arange(22),
        "val_members"   : np.arange(22, 25),
        "test_members"  : np.arange(25, 30),
        "time_range"    : "Tier1",

        "CNN_blocks"    : [2],
        "CNN_filters"   : [32],
        "CNN_kernals"   : [3],
        "CNN_strides"   : [1],

        "encoding_nodes" : [100,100,],
        "code_nodes"     : 100,
        "activation"     : "tanh",
        "variational_loss": 0.0001
    },

    "tos_feature_forced" : {
        "seed"          : 0 ,
        "learn_rate" : 0.001,
        "patience"      : 100,
        "batch_size"    : 64,
        "max_epochs"    : 2000,
        "metric"        : "",
        
        "input_std_method"    : "feature", #self, feature, overall
        "output_std_method"   : "feature", # feature, overall
        "input_variable": ["pr", "tas", "psl", "tos"],
        "target_variable": "tos",
        "target_component": "forced", #forced or internal
        "train_models"  : ["CESM2", "MIROC6", "CanESM5", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "val_models"    : ["CESM2", "MIROC6", "CanESM5", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "test_models"   : ["CESM2", "MIROC6", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "train_members" : np.arange(22),
        "val_members"   : np.arange(22, 25),
        "test_members"  : np.arange(25, 30),
        "time_range"    : "Tier1",

        "CNN_blocks"    : [2],
        "CNN_filters"   : [32],
        "CNN_kernals"   : [3],
        "CNN_strides"   : [1],

        "encoding_nodes" : [100,100,],
        "code_nodes"     : 100,
        "activation"     : "tanh",
        "variational_loss": 0.0001
    },

    "tos_feature_internal" : {
        "seed"          : 0 ,
        "learn_rate" : 0.005,
        "patience"      : 10,
        "batch_size"    : 64,
        "max_epochs"    : 2000,
        "metric"        : "",
        
        "input_std_method"    : "feature", #self, feature, overall
        "output_std_method"   : "feature", # feature, overall
        "input_variable": ["pr", "tas", "psl", "tos"],
        "target_variable": "tos",
        "target_component": "internal", #forced or internal
        "train_models"  : ["CESM2", "MIROC6", "CanESM5", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "val_models"    : ["CESM2", "MIROC6", "CanESM5", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "test_models"   : ["CESM2", "MIROC6", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "train_members" : np.arange(22),
        "val_members"   : np.arange(22, 25),
        "test_members"  : np.arange(25, 30),
        "time_range"    : "Tier1",

        "CNN_blocks"    : [2],
        "CNN_filters"   : [32],
        "CNN_kernals"   : [3],
        "CNN_strides"   : [1],

        "encoding_nodes" : [100,100,],
        "code_nodes"     : 0,
        "activation"     : "tanh",
        "variational_loss": 0.0001
    },

    "psl_exp_internal" : {
        "seed"          : 0 ,
        "learn_rate" : 0.001,
        "patience"      : 100,
        "batch_size"    : 64,
        "max_epochs"    : 2000,
        "metric"        : "",
        
        "input_std_method"    : "self", #self, feature, overall
        "output_std_method"   : "feature", # feature, overall
        "input_variable": ["pr", "tas", "psl", "tos"],
        "target_variable": "psl",
        "target_component": "internal", #forced or internal
        "train_models"  : ["CESM2", "MIROC6", "CanESM5", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "val_models"    : ["CESM2", "MIROC6", "CanESM5", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "test_models"   : ["CESM2", "MIROC6", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "train_members" : np.arange(22),
        "val_members"   : np.arange(22, 25),
        "test_members"  : np.arange(25, 30),
        "time_range"    : "Tier1",

        "CNN_blocks"    : [2],
        "CNN_filters"   : [32],
        "CNN_kernals"   : [3],
        "CNN_strides"   : [1],

        "encoding_nodes" : [100,100,],
        "code_nodes"     : 3,
        "activation"     : "tanh",
        "variational_loss": 0.0001
    },

    "psl_exp_forced" : {
        "seed"          : 0 ,
        "learn_rate" : 0.001,
        "patience"      : 100,
        "batch_size"    : 64,
        "max_epochs"    : 2000,
        "metric"        : "",
        
        "input_std_method"    : "self", #self, feature, overall
        "output_std_method"   : "feature", # feature, overall
        "input_variable": ["pr", "tas", "psl", "tos"],
        "target_variable": "psl",
        "target_component": "forced", #forced or internal
        "train_models"  : ["CESM2", "MIROC6", "CanESM5", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "val_models"    : ["CESM2", "MIROC6", "CanESM5", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "test_models"   : ["CESM2", "MIROC6", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "train_members" : np.arange(22),
        "val_members"   : np.arange(22, 25),
        "test_members"  : np.arange(25, 30),
        "time_range"    : "Tier1",

        "CNN_blocks"    : [2],
        "CNN_filters"   : [32],
        "CNN_kernals"   : [3],
        "CNN_strides"   : [1],

        "encoding_nodes" : [100,100,],
        "code_nodes"     : 3,
        "activation"     : "tanh",
        "variational_loss": 0.0001
    },

    "tas_exp_internal" : {
        "seed"          : 0 ,
        "learn_rate" : 0.001,
        "patience"      : 100,
        "batch_size"    : 64,
        "max_epochs"    : 2000,
        "metric"        : "",
        
        "input_std_method"    : "self", #self, feature, overall
        "output_std_method"   : "feature", # feature, overall
        "input_variable": ["pr", "tas", "psl", "tos"],
        "target_variable": "tas",
        "target_component": "internal", #forced or internal
        "train_models"  : ["CESM2", "MIROC6", "CanESM5", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "val_models"    : ["CESM2", "MIROC6", "CanESM5", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "test_models"   : ["CESM2", "MIROC6", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "train_members" : np.arange(22),
        "val_members"   : np.arange(22, 25),
        "test_members"  : np.arange(25, 30),
        "time_range"    : "Tier1",

        "CNN_blocks"    : [2],
        "CNN_filters"   : [32],
        "CNN_kernals"   : [3],
        "CNN_strides"   : [1],

        "encoding_nodes" : [100,100,],
        "code_nodes"     : 100,
        "activation"     : "tanh",
        "variational_loss": 0.0001
    },

    "tas_exp_forced" : {
        "seed"          : 0 ,
        "learn_rate" : 0.001,
        "patience"      : 100,
        "batch_size"    : 64,
        "max_epochs"    : 2000,
        "metric"        : "",
        
        "input_std_method"    : "self", #self, feature, overall
        "output_std_method"   : "feature", # feature, overall
        "input_variable": ["pr", "tas", "psl", "tos"],
        "target_variable": "tas",
        "target_component": "forced", #forced or internal
        "train_models"  : ["CESM2", "MIROC6", "CanESM5", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "val_models"    : ["CESM2", "MIROC6", "CanESM5", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "test_models"   : ["CESM2", "MIROC6", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "train_members" : np.arange(22),
        "val_members"   : np.arange(22, 25),
        "test_members"  : np.arange(25, 30),
        "time_range"    : "Tier1",

        "CNN_blocks"    : [2],
        "CNN_filters"   : [32],
        "CNN_kernals"   : [3],
        "CNN_strides"   : [1],

        "encoding_nodes" : [100,100,],
        "code_nodes"     : 100,
        "activation"     : "tanh",
        "variational_loss": 0.0001
    },

    "zmta_exp_forced" : {
        "seed"          : 0 ,
        "learn_rate" : 0.001,
        "patience"      : 100,
        "batch_size"    : 64,
        "max_epochs"    : 2000,
        "metric"        : "",
        
        "input_std_method"    : "self", #self, feature, overall
        "output_std_method"   : "feature", # feature, overall
        "input_variable": ["zmta", "pr", "tas", "psl", "tos"],
        "target_variable": "zmta",
        "target_component": "forced", #forced or internal
        "train_models"  : ["CESM2", "MIROC6", "CanESM5", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "val_models"    : ["CESM2", "MIROC6", "CanESM5", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "test_models"   : ["CESM2", "MIROC6", "MPI-ESM1-2-LR", "MIROC-ES2L",],
        "train_members" : np.arange(22),
        "val_members"   : np.arange(22, 25),
        "test_members"  : np.arange(25, 30),
        "time_range"    : "Tier1",

        "CNN_blocks"    : [2],
        "CNN_filters"   : [32],
        "CNN_kernals"   : [3],
        "CNN_strides"   : [1],

        "encoding_nodes" : [100,100,],
        "code_nodes"     : 100,
        "activation"     : "tanh",
        "variational_loss": 0.0001
    },
}
