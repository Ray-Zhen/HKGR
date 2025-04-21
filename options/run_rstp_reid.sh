#!/bin/bash
DATASET_NAME="rstp_reid"

CUDA_VISIBLE_DEVICES=1 \
python train.py \
--name rstp_full_tse \
--img_aug \
--root_dir 'path_of_dataset' \
--batch_size 64 \
--MLM \
--mlm_type 'obj+attr+rel' \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+mlm+id' \
--num_epoch 60 \
--num_workers 12 \
--lr 1e-5 \
--temperature 0.02 \
--weight_decay 1e-2 \
