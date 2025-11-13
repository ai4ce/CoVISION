#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node a100:2 --constraint=gpu_a100_80gb
#SBATCH --partition=viprgs
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mem=250GB
#SBATCH --job-name=MultiLoc
#SBATCH --mail-type=END
#SBATCH --mail-user=ndang@clemson.edu

module load cuda

cd /project/bli4/autoai/nobel/chao_mvdust3r_exp_covision/mvdust3r_vpr_bce_mask_combined
source activate mvdp

bash scripts/train_mvd_combined.sh 

# srun torchrun --master_port=12345 --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 train.py \
# --train_dataset " \
# 4300 @ MVDataset(split='all', ROOT='trajectories/covision_train', aug_crop=16, mask_bg='rand', resolution=224, transform=ColorJitter, num_views=7, num_render_views=2, random_order = True, random_render_order = False, dps_name = 'dataset_train.json', random_nv_nr = [[7,2]]) + 10000 @ HM3DDataset(split='all', ROOT='trajectories/HM3D_train', aug_crop=16, mask_bg='rand', resolution=224, transform=ColorJitter, num_views=7, num_render_views=2, random_order = True, random_render_order = False, dps_name = 'dataset_train.json', random_nv_nr = [[7,2]])" \
# --test_dataset " \
# 200 @ MVDataset(n_all=1000, split='all', ROOT='trajectories/covision_test', resolution=224, seed=777, fix_order=True, num_views=7, num_render_views=2, render_start = 8, dps_name = 'dataset_test.json', tb_name = 'mp3d_tdf_2_testFull_7_2', n_ref = 1) + 300 @ HM3DDataset(n_all=1000, split='all', ROOT='trajectories/HM3D_test', resolution=224, seed=777, fix_order=True, num_views=7, num_render_views=2, render_start = 8, dps_name = 'dataset_test.json', tb_name = 'mp3d_tdf_2_testFull_7_2', n_ref = 1)" \
# --model "AsymmetricCroCo3DStereoMultiView(pos_embed='RoPE100', img_size=(224, 224), head_type='linearvpr', output_mode='feat', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, 1e9), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, GS = True, sh_degree=0, pts_head_config = {'skip':True})" \
# --train_criterion "MaskBCELoss()" \
# --test_criterion "MaskBCELoss()" \
# --pretrained "checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth" \
# --lr 0.00015 --min_lr 0.000001 --warmup_epochs 1 --epochs 100 --batch_size 1 --accum_iter 2 --save_freq 1 --keep_freq 5 --eval_freq 5 --num_workers 0 \
# --output_dir outputs/MVD_mask_full_ref_bk
