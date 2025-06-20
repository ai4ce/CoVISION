#!/bin/bash

source $SCRATCH/environments/miniconda3/etc/profile.d/conda.sh
export PATH=$SCRATCH/environments/miniconda3/bin:$PATH
export PATH=/usr/local/cuda-11/bin:$PATH
export PYTHONPATH=$SCRATCH/environments/miniconda3/bin:$PYTHONPATH
export PYTHONPATH=$SCRATCH/habitat-sim/src_python:$PYTHONPATH
#export OPENGL_INCLUDE_DIR=/usr/include
#export OPENGL_opengl_LIBRARY=/usr/lib64/libGL.so
#export OPENGL_glx_LIBRARY=/usr/lib64/libGLX.so
export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64:$LD_LIBRARY_PATH
