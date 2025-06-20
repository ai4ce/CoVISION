#!/bin/bash


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --mem=4GB
#SBATCH --gres=gpu
#SBATCH --job-name=habitat_dataset_gen
#SBATCH --mail-type=END
#SBATCH --mail-user=nd2100@nyu.edu


module purge


singularity exec --nv \
        --bind /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
        --overlay /scratch/$USER/environments/habitat.ext3:rw \
        /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
        /bin/bash -c "source /ext3/env.sh; conda activate habitat; python statistics.py"


