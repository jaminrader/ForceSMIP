#!/bin/bash

##SBATCH --nodelist=mal-node2
##SBATCH --partition=mal_all
#SBATCH --partition=mal_all
#SBATCH --nodelist=mal-node2
#SBATCH --cpus-per-task=5

# Some more handy options. For a full list or explanation of an optional https://slurm.schedmd.com/sbatch.html
#SBATCH --output=monmaxtasmax_iv.txt
#SBATCH --job-name=monmaxtasmax_iv
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jaminr@colostate.edu

experiment_name="internal_test_standard"
var="tos" # tos, tas, pr, psl, zmta, monmaxtasmax, monmintasmin, monmaxpr, mrso

# python3 run_experiment_new.py "${experiment_name}_${var}_annual"
python3 run_experiment_new.py "${experiment_name}_${var}_1"
python3 run_experiment_new.py "${experiment_name}_${var}_2"
python3 run_experiment_new.py "${experiment_name}_${var}_3"
python3 run_experiment_new.py "${experiment_name}_${var}_4"
python3 run_experiment_new.py "${experiment_name}_${var}_5"
python3 run_experiment_new.py "${experiment_name}_${var}_6"
python3 run_experiment_new.py "${experiment_name}_${var}_7"
python3 run_experiment_new.py "${experiment_name}_${var}_8"
python3 run_experiment_new.py "${experiment_name}_${var}_9"
python3 run_experiment_new.py "${experiment_name}_${var}_10"
python3 run_experiment_new.py "${experiment_name}_${var}_11"
python3 run_experiment_new.py "${experiment_name}_${var}_12"


