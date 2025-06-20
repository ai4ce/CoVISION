META_INTERNAL=False python train.py \
--train_dataset " \
8000 @ MVDataset(split='all', ROOT='trajectories/covision_train', aug_crop=16, mask_bg='rand', resolution=224, transform=ColorJitter, num_views=12, num_render_views=2, random_order = True, random_render_order = False, dps_name = 'dataset_train.json', random_nv_nr = [[12,2]])" \
--test_dataset " \
MVDataset(n_all=1000, split='all', ROOT='trajectories/covision_test', resolution=224, seed=777, fix_order=True, num_views=30, num_render_views=6, render_start = 24, dps_name = 'dataset_test.json', tb_name = 'mp3d_tdf_2_testFull_10_2', n_ref = 1)" \
--model "AsymmetricCroCo3DStereoMultiView(pos_embed='RoPE100', img_size=(224, 224), head_type='linearvpr', output_mode='feat', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, 1e9), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, GS = True, sh_degree=0, pts_head_config = {'skip':True})" \
--train_criterion "BCELoss()" \
--test_criterion "BCELoss()" \
--pretrained "checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth" \
--lr 0.00015 --min_lr 0.000001 --warmup_epochs 1 --epochs 100 --batch_size 2 --accum_iter 2 --save_freq 1 --keep_freq 5 --eval_freq 5 --num_workers 0 \
--output_dir outputs2/MVD



# --train_criterion "GSRenderLoss(L21, norm_mode='avg_dis', mv = True, scale_scaled = True, use_gt_pcd = False, lpips_coeff = 1.0, rgb_coeff = 1.0, use_img_rgb = True, local_loss_coeff=0.0) + ConfLoss(Regr3D(L21, norm_mode='avg_dis', mv = True), alpha=0.2)" \
# --test_criterion "GSRenderLoss(L21, norm_mode='avg_dis', mv = True, render_included = True, scale_scaled = True, use_img_rgb = True, local_loss_coeff=0.0) + Regr3D_ScaleShiftAllInv(L21, gt_scale=True, mv = True)" \