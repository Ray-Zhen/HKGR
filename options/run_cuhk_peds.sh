#!/bin/bash
DATASET_NAME="cuhk_peds"

CUDA_VISIBLE_DEVICES=3 \
python train.py \
--name cuhk_Lga+Lggr \
--img_aug \
--root_dir 'path_of_dataset' \
--batch_size 64 \
--MLM \
--mlm_type 'obj+attr+rel' \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+mlm' \
--num_epoch 60 \
--num_workers 12 \
--lr 1e-5 \
--temperature 0.01 \
--weight_decay 1e-3 \
