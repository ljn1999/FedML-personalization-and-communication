import argparse
import logging
import os
import string
import sys

import numpy as np
import torch
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.standalone.hierarchical_fl.trainer import Trainer
from fedml_api.standalone.fedavg.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from fedml_api.standalone.fedavg.my_model_trainer_nwp import MyModelTrainer as MyModelTrainerNWP
from fedml_api.standalone.fedavg.my_model_trainer_tag_prediction import MyModelTrainer as MyModelTrainerTAG
from fedml_experiments.standalone.fedavg.main_fedavg import add_args, load_data, create_model

if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = add_args(argparse.ArgumentParser(description='GroupFedAvg-standalone'))
    parser.add_argument('--group_method', type=str, default='random', metavar='N', help='how clients should be grouped')
    parser.add_argument('--group_num', type=int, default=1, metavar='N', help='the number of groups')
    parser.add_argument('--global_comm_round', type=int, default=10, help='the number of global communications')
    parser.add_argument('--group_comm_round', type=int, default=10,
                        help='the number of group communications within a global communication')
    parser.add_argument('--personalize', type=int, default=0)
    parser.add_argument('--communication', type=int, default=0)
    parser.add_argument('--writers', type=int, nargs="+", default=[])
    args = parser.parse_args()
    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)

    wandb.init(
        project="fedml",
        name="GroupFedAvg-{}-{}-{}-{}-{}".format(args.group_method, args.group_num, args.global_comm_round, args.group_comm_round, args.epochs),
        config=args
    )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    np.random.seed(0)
    torch.manual_seed(10)

    # load data
    dataset = load_data(args, args.dataset)

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(args, model_name=args.model, output_dim=dataset[7])
    logging.info(model)

    #trainer = Trainer(dataset, model, device, args)

    if args.dataset == "stackoverflow_lr":
        model_trainer = MyModelTrainerTAG(model)
    elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
        model_trainer = MyModelTrainerNWP(model)
    else: # default model trainer is for classification problem
        model_trainer = MyModelTrainerCLS(model)
    
    if args.personalize == 1:
        personalize = True
    else:
        personalize = False

    if args.communication == 1:
        communication = True
    else:
        communication = False

    trainer = Trainer(dataset, device, args, model_trainer, personalize, communication)
    trainer.train(personalize, communication)
