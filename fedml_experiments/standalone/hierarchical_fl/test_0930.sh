#!/usr/bin/env bash

GPU=0

CLIENT_NUM=10

WORKER_NUM=10

BATCH_SIZE=128

DATASET='cifar10'

DATA_PATH='./../../../data/cifar10'

MODEL='resnet56'

DISTRIBUTION='hetero'

LR=0.001

OPT=adam

GROUP_METHOD='random'

GROUP_NUM=1

GLOBAL_COMM_ROUND=10

GROUP_COMM_ROUND=10

EPOCH=2

python3 ./main.py \
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
--epochs $EPOCH
