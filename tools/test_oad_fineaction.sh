PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# srun -p Gvlab-S1 -N1 -n2 --pty --gres=gpu:2 --job-name=test_thomus --quotatype=auto --cpus-per-task=12 \
# python -u -m main_oad \
srun -p Gvlab-S1 -N1 -n2  --preempt --pty --gres=gpu:2 --job-name=test_fineaction --quotatype=spot --cpus-per-task=16 \
python -u -m main_oad \
    --cfg configs/ovoador/test_oad_fineaction.yml \
    --resume /mnt/cache/heyinan/00_zqs/data/exps/anet_enc32_3-layers_txt.enc-unfixed_lsxattn-0.1_multilabel-maskloss/ovoador_bs256x1/thomus_best_map.pth \
    # --resume /mnt/petrelfs/heyinan/00_zqs/data/exps/n2w_enc32_adapter_e30_matchingloss_lre5_withbg/ovoador_bs256x1/best_map_3_23.pth \
    # --resume  /mnt/petrelfs/heyinan/00_zqs/data/exps/n5k_enc32_T-adapter_e30_matchingloss_lre5_withbg/ovoador_bs256x1/best_map.pth \
    # --resume /mnt/petrelfs/heyinan/00_zqs/data/exps/n2w_enc32_vl_adapter_e30_matchingloss_lre5_withbg/ovoador_bs256x1/best_map.pth \
    # --resume /mnt/petrelfs/heyinan/00_zqs/data/exps/n2w_enc32_adapter_e30_multilabel_matchingloss_lre5_withbg/ovoador_bs256x1/best_map.pth \
    # --resume /mnt/petrelfs/heyinan/00_zqs/data/exps/n2w_enc24_nz8_adapter_e30_multilabel_actionloss_lre5/ovoador_bs256x1/best_map.pth \ 
    # --resume /mnt/petrelfs/heyinan/00_zqs/data/exps/n2w_enc24_clip_txtenc_multilabel_actionloss_group_8_2_lre5/ovoador_bs256x1/best_map.pth \ 
    # --resume /mnt/petrelfs/heyinan/00_zqs/data/exps/n2w_enc32_e5_multilabel_actionloss_lre5_withbg/ovoador_bs256x1/best_map.pth \
    # --resume /mnt/petrelfs/heyinan/00_zqs/data/exps/n2w_enc32_nz2_clip_txtenc_multilabel_matching_group_8_2_lre5/ovoador_bs256x1/best_map.pth \
    # --resume /mnt/petrelfs/heyinan/00_zqs/data/exps/n10w_enc32_clip_txtenc_maskloss_group_8_2_lre5_clip/ovoador_bs256x1/checkpoint.pth \
    # --resume /mnt/petrelfs/heyinan/00_zqs/data/exps/n10w_enc32_maskloss_stages3_group8-2-2_lre5/ovoador_bs256x1/checkpoint.pth \
    # --resume /mnt/petrelfs/heyinan/00_zqs/data/exps/n10w_enc32_maskloss_avg_val/ovoador_bs256x1/checkpoint.pth \
    # --resume /mnt/petrelfs/heyinan/00_zqs/data/exps/n5000_b256_d512_v0/ovoador_bs256x1/ckpt_epoch_10.pth \
    # --resume /mnt/petrelfs/heyinan/00_zqs/data/exps/n2000_b256_d512_v0/ovoador_bs256x1/checkpoint.pth \
    # --resume /mnt/petrelfs/heyinan/00_zqs/data/exps/n2000_b256_d512_v0/ovoador_bs256x1/ckpt_epoch_10.pth \

# n2w_enc32_adapter_e30_matchingloss_lre5_withbg/ovoador_bs256x1/best_map_3_23.pth
#  mAPs: {'thomus': 29.138653393914726, 'tvseries': 4.616162097587261, 'epic': 1.0954657218393076, 'anet': 68.2349644802404}, 
#  cmAPs: {'thomus': 94.41475523745926, 'tvseries': 69.86103258001887, 'epic': 40.56162470173007, 'anet': 98.76187696138624}.

