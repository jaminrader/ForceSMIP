import numpy as np

def get_experiment(exp_name, base_exp_name=None, settings_overwrite=None):

    if settings_overwrite is None:
        settings = get_experiment_settings(exp_name)
    else:
        settings = settings_overwrite[exp_name]
    settings["exp_name"] = exp_name

    if base_exp_name is not None:
        settings["base_exp_name"] = base_exp_name

    if "ignore_smooth_warning" in list(settings.keys()) and settings["ignore_smooth_warning"] == True:
        print('IGNORING SMOOTHING WARNINGS')
    else:
        assert settings["lead_time"] >= 0, f"lead_time must be non-negative."
        assert settings["smooth_len_input"] >= 0, f"input smoothing must be non-negative."
        assert settings["smooth_len_output"] <= 0, f"output smoothing must be non-positive."

    return settings

def get_experiment_settings(exp_name):

    return experiments[exp_name]

experiments = {
    "exp_template" : {
        #Ran by: 
        #Metrics:
        #Notes:
        "seed"        : 0 ,
        "learn_rate"  : .001,
        "patience"    : 50,
        "batch_size"  : 64,
        "max_epochs"  : 5_000,
        "metric"      : "",
        
        "std_method"  : "",
        "variable"    : "",
        "nmodels"     : [],
        "nmembers"    : [],
        "time_range"  : 0,

        "CNN_blocks"  : [2],
        "CNN_filters" : [32],
        "CNN_kernals" : [3],
        "CNN_strides" : [1],

        "encoding_nodes" : [20, 20, 10],
        "code_nodes"     : 1,
        "activation"     : "linear",
        "variational_loss": .001
    },
}
