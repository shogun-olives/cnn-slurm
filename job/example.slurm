#!/bin/bash
#SBATCH --job-name=example
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus_per_task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=2:0:00
#SBATCH --output=./model/example/example_%j.log

export GIT_PYTHON_REFRESH=quiet
conda init
source ~/.bashrc
conda activate yolo_v5

python ../main.py --dest ./model/example