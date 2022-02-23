#!/bin/bash
GPU=0

CLIENT_NUM=3 # useless for utkface

WORKER_NUM=6

BATCH_SIZE=128 # useless for utkface

DATASET='utkface'

DATA_PATH='./../../../data/UTKFace/UTKFace'

MODEL='cnn'

DISTRIBUTION='hetero'

LR=0.005

OPT=adam

GROUP_METHOD='random'

GROUP_NUM=2

GLOBAL_COMM_ROUND=2

GROUP_COMM_ROUND=3

EPOCH=10

python ./main.py \
--gpu $GPU \
--client_num_in_total $CLIENT_NUM \
--client_num_per_round $WORKER_NUM \
--batch_size $BATCH_SIZE \
--dataset $DATASET \
--data_dir $DATA_PATH \
--model $MODEL \
--partition_method $DISTRIBUTION  \
--lr $LR \
--client_optimizer $OPT \
--group_method $GROUP_METHOD \
--group_num $GROUP_NUM \
--global_comm_round $GLOBAL_COMM_ROUND \
--group_comm_round $GROUP_COMM_ROUND \
--epochs $EPOCH \
--personalize 1 \