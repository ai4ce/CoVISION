## Installation

1. Install the virtual environment under anaconda.

```bash
./install.sh
```

(version of pytorch and pytorch3d should be changed if you need other CUDA version.)

2. (Optional for faster runtime) Compile the cuda kernels for RoPE (the same as [DUSt3R and Croco](https://github.com/naver/dust3r?tab=readme-ov-file#installation))

```bash
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```

## Checkpoints

Please download CoVis checkpoint (To be released) to the folder outputs/ for inference. You can also download the DUSt3R checkpoint 'DUSt3R_ViTLarge_BaseDecoder_224_linear.pth' from github[https://github.com/naver/dust3r] as initilization for the training process.

|     Name    | Description |
|-------------|-------------|
| checkpoint-last.pth | Covis checkpoint |


## Data

You can download our dataset for Co-VisiON by following steps from the main branch or at [link](https://huggingface.co/datasets/ai4ce/CoVISION/tree/main).

## Evaluation

You can choose to run any of the three scripts for evaluation: [calculate_IOU_save_csv_Gibson.py](./calculate_IOU_save_csv_Gibson.py), [calculate_IOU_save_csv_HM3D.py](./calculate_IOU_save_csv_HM3D.py) or calculate_IOU_save_csv_mask.py((./calculate_IOU_save_csv_mask.py)):

## Training

1. First download our CoVisiON dataset.
2. Download .json files and put them under trajectories covision_train, covision_test, HM3D_train, HM3D_test from [link](TBD)
2. Run the trajectories/compute_gvgg_traj_multi_mask2.py and trajectories/compute_hvgg_traj_multi_mask3.py
3. Download the DUSt3R checkpoint file 'DUSt3R_ViTLarge_BaseDecoder_224_linear.pth'
4. For training run the bash file scripts/train_mvd_combined.sh


## Citation

```bibtex
@inproceedings{chen2025co,
  title={Co-VisiON: Co-Visibility ReasONing on Sparse Image Sets of Indoor Scenes},
  author={Chen, Chao and Dang, Nobel and Zhang, Juexiao and Sun, Wenkai and Zheng, Pengfei and He, Xuhang and Ye, Yimeng and Zhang, Jiasheng and Srinivas, Taarun and Feng, Chen},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4802--4812},
  year={2025}
}
```

## License

We use [CC BY-NC 4.0]

## Acknowledgement

Many thanks to great repositories from:
- [MV-DUSt3R+](https://github.com/facebookresearch/mvdust3r)
- [DUSt3R](https://github.com/naver/dust3r)