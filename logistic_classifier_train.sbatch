#!/bin/bash
#SBATCH -n 1
#SBATCH --gres=gpu:1

# Clear the environment from any previously loaded modules
#module purge > /dev/null 2>&1

# Load the module environment suitable for the job
#module load foss/2019a

# And finally run the job​

source /home/${USER}/.bashrc;
source activate tf1.15-env; 
nvidia-smi; 
python stylegan_generation/logistic_classfier_all_race.py  -e  ~/dataset/logistic-classifier-models -p  ~/dataset/stylegan-generated-vanilla
