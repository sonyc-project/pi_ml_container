#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=rpi-test-ust-tflite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=1-0
#SBATCH --mail-type=FAIL
#SBATCH --output="rpi-test-ust-tflite-%j.out"

source ~/.bashrc

module purge
module load cudnn/10.0v7.4.2.24

echo 'Using tensorflow 2.0 GPU'
source activate l3embedding-tf-2-gpu

python $SCRATCH/pi_ml_container/test_demo_model_rpi_ust.py

