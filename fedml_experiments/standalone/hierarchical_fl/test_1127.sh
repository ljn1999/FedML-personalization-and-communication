#!/usr/bin/env bash

GPU=0

CLIENT_NUM=3

WORKER_NUM=2

BATCH_SIZE=128

DATASET='femnist'

DATA_PATH='./../../../data/FederatedEMNIST/datasets'

MODEL='cnn'

DISTRIBUTION='hetero'

LR=0.01

OPT=adam

GROUP_METHOD='random'

GROUP_NUM=3

GLOBAL_COMM_ROUND=3

GROUP_COMM_ROUND=5

EPOCH=10

SAMPLE_BASE_NUM=0.9

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
--sample_base_num $SAMPLE_BASE_NUM
