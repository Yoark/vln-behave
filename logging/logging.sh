# This bash file demonstrate how to run the code for logging.
# pay attention to --ablate_name, --dataset, --test_type, --store_final_logits 

features=vitbase_clip
ft_dim=512

ngpus=1
seed=0

outdir=/usr/src/data/unitVLN/datasets/RxR/trained_models

flag="--root_dir /usr/src/data/unitVLN/datasets
      --output_dir ${outdir}
      
      --dataset rxr
      --ablate_name 7_hop_intervene_dataset 7_hop_no_intervene_dataset 8_hop_intervene_dataset 8_hop_no_intervene_dataset
      --test_type type2
      --store_final_logits

      --ob_type pano
      --no_lang_ca

      --world_size ${ngpus}
      --seed ${seed}
      
      --fix_lang_embedding
      --fix_hist_embedding

      --num_l_layers 9
      --num_x_layers 4
      --hist_enc_pano
      --hist_pano_num_layers 2

      --features ${features}
      --feedback sample

      --max_action_len 20
      --max_instr_len 250
      --batch_size 16

      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --lr 1e-5
      --iters 200000
      --log_every 2000
      --optim adamW

      --ml_weight 0.2
      --featdropout 0.4
      --dropout 0.5"

# inference/
CUDA_VISIBLE_DEVICES=0 python3 main.py $flag  \
      --resume_file /usr/src/data/unitVLN/datasets/RxR/trained_models/vitbase.clip-finetune/best_val_unseen \
      --test --submit
      
# train
# CUDA_VISIBLE_DEVICES='5' python r2r/main.py $flag  \
#       --bert_ckpt_file /usr/src/data/unitVLN/datasets/RxR/trained_models/vitbase.clip-pretrain/model_step_200000.pt \
#       --eval_first

