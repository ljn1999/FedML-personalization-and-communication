#!/bin/bash
#SBATCH --account=def-ssanner
#SBATCH --mail-user=nini.li@mail.utoronto.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=32g
#SBATCH --time=12:00:0

GPU=0

CLIENT_NUM=3 # useless for femnist

WORKER_NUM=30

BATCH_SIZE=128 # useless for femnist

DATASET='femnist'

DATA_PATH='./../../../data/FederatedEMNIST/datasets'

MODEL='cnn'

DISTRIBUTION='hetero'

LR=0.01

OPT=adam

GROUP_METHOD='random'

GROUP_NUM=3

GLOBAL_COMM_ROUND=4

GROUP_COMM_ROUND=3

EPOCH=5

QUANTIZATION_MODE='8-bit'

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
--communication 1 \
--quantization_mode $QUANTIZATION_MODE
