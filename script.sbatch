#!/bin/bash
#SBATCH -n 1
#SBATCH --gres=gpu:4

# Clear the environment from any previously loaded modules
#module purge > /dev/null 2>&1

# Load the module environment suitable for the job
#module load foss/2019a

# And finally run the job​

source /home/${USER}/.bashrc;
source activate tf2-env; 
nvidia-smi; 
python UTK_face/preprocessing/train_vggface_race_classifier.py -e 50 -p ~/dataset/UTK-FACE-preprocessed-dlib -d ~/dataset/UTK-FACE-preprocessed-models
#srun --mem-per-gpu=10G --cpus-per-gpu=4 --gpus=1 python UTK_face/preprocessing/train_vggface_race_classifier.py -p ~/dataset/UTK-FACE-preprocessed-dlib -d ~/dataset/UTK-FACE-preprocessed-models
