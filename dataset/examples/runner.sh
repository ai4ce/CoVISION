#!/bin/bash
#SBATCH --nodes=1                        # requests 1 compute server
#SBATCH --ntasks-per-node=1              # runs 1 task on each server
#SBATCH --cpus-per-task=20                # uses 4 compute cores per task
#SBATCH --time=6:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=spatial-reasoning
#SBATCH --output=spatial-reasoning.out
#SBATCH --gres=gpu:rtx8000:1  ## To request specific GPU (v100 or rtx8000)

singularity exec --nv \
      --bind /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
        --overlay /scratch/NET_ID/environments/habitat.ext3:ro \
        /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
        /bin/bash -c "source /ext3/env.sh; conda activate habitat; cd /scratch/NET_ID/spatial-reasoning/dataset/examples; python ./algo.py; exit"