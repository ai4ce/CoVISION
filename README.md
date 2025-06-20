# Spatial_reasoning

To run Spatial reasoning with coviriance images, we need to build a modifed 
version of habitat-sim from source. Make sure to follow the following steps
to make it work on HPC clusters.

## Running algorithm.py

### algo.py setup

### 1.Download the code

After logging into HPC clusters,
Clone both Spatial_reasoning repository and Habitat-sim repository:
```
cd /$SCRATCH
git clone https://github.com/ai4ce/Spatial_reasoning.git
git clone --branch stable https://github.com/facebookresearch/habitat-sim.git
```
Then change to Spatial_reasoning and checkout to CameraReady branch and copy
out modified habitat-sim srouce file to habitat-sim repo.
```
cd Spatial_reasoning
git checkout CameraReady
sh copy.sh
```

### 2. Download necessary scene files.
```
mkdir /$SCRATCH/spatial-reasoning/dataset/examples/gibson
```
Download the gibson file .glb and .navmesh files, along with the metadata [JSON file](https://raw.githubusercontent.com/StanfordVL/GibsonEnv/master/gibson/data/data.json). For docs see here -> [Documentation](https://github.com/StanfordVL/GibsonEnv/blob/master/gibson/data/README.md)
```
cd /$SCRATCH/spatial-reasoning/dataset/examples/gibson
curl -o gibson_floors.json https://raw.githubusercontent.com/StanfordVL/GibsonEnv/master/gibson/data/data.json
curl -O https://dl.fbaipublicfiles.com/habitat/data/scene_datasets/gibson_habitat.zip
unzip gibson_habitat.zip
```

### 3. Set up environment

1. Get GPU node and setup container.
    Researve a GPU, if a100 is occupied try replace a100 with rtx8000.
    ```
    srun --gres=gpu:a100:1 --cpus-per-task=2 --time=1:59:59 --mem=32GB --pty --account=pr_110_tandon_priority /bin/bash
    ```
    Launch the Singularity container in read/write mode (with the :rw flag).
    ```
    singularity exec --nv --bind /usr/share/glvnd/egl_vendor.d/10_nvidia.json /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash
    ```

2. Setup environment.
    Create a directory for the environment and copy env files to the directory.
    ```
    mkdir /$SCRATCH/environments
    cd /$SCRATCH/environments
    cp /$SCRATCH/Spatial_reasoning/env.sh ./
    ```

    Now download minicoda3 and setup the environment.
    ```
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    sh Miniconda3-latest-Linux-x86_64.sh -b -p ./miniconda3
    source env.sh
    ```

    Create a new conda environment for habitat.
    ```
    conda create -n habitat python=3.9 cmake=3.14.0 -y
    conda activate habitat
    ```
    Install required dependencies.
    ```
    conda install -c pytorch pytorch
    conda install -c conda-forge open3d
    pip install numpy==1.26.4 numpy-quaternion==2023.0.4
    pip install scikit-learn
    pip install opencv-python
    pip install glfw PyOpenGL torchvision
    ```

3. Build habitat-sim from source.
    Go to habitat-sim directory and install its dependencies.
    ```
    cd /$SCRATCH/habitat-sim
    pip install -r requirements.txt
    ```
    Now build habitat-sim from source.
    ```
    ./build.sh --headless --with-cuda --bullet
    ```

### 4. Run the code
    Make sure you are on GPU node and inside singularity container as we have
    done in step 3-1, also make sure environment has been setup correctly in 
    step 3-2.

    Run our code! Mind that you may want to change your dataset path accordingly.
    ```
    cd /$SCRATCH/Spatial_reasoning/dataset/examples
    python algo.py
    ```

### 5. Handler to draw covariant images.
    The API I provided is
    ```
    def get_covariant_image(self, scene_name, id1, position1, rotation1,
        id2, position2, rotation2):
    ```
    inside `test_runner.py` where we can call runner.get_covariant_image()
    anytime to generate covariant images. `scene_name`, `id1`, `id2` are scene
    name string and id of 2 images. `position1, rotation1, position2, rotation2`
    have to be numpy arrays that represent position and rotation of 2 images.

    Nothing will be returned, the images will be written to the path
    `data_folder/More_vis/covariant_images`.

### 6. Check your images in data_folder/More_viscovariant_images folders
