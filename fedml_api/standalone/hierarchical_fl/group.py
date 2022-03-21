import logging
import math
import numpy as np
import torch
from collections import OrderedDict
from fedml_api.standalone.hierarchical_fl.client import Client
from fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI

class Group(FedAvgAPI):

    def __init__(self, idx, total_client_indexes, train_data_local_dict, test_data_local_dict, train_data_local_num_dict, args, device, model_trainer):
        self.idx = idx
        self.args = args
        self.device = device
        self.client_dict = {}
        self.train_data_local_num_dict = train_data_local_num_dict
        for client_idx in total_client_indexes:
            self.client_dict[client_idx] = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], args, device, model_trainer)

    def get_sample_number(self, sampled_client_indexes):
        self.group_sample_number = 0
        for client_idx in sampled_client_indexes:
            self.group_sample_number += self.train_data_local_num_dict[client_idx]
        return self.group_sample_number

    def train(self, global_round_idx, w, client_indexes, sample_base_num):
        client_list = [self.client_dict[client_idx] for client_idx in client_indexes]
        w_group = w
        w_group_list = []
        for group_round_idx in range(self.args.group_comm_round):
            logging.info("Group ID : {} / Group Communication Round : {}".format(self.idx, group_round_idx))
            w_locals_dict = {}

            # train each client
            client_to_gradient_dict = {}
            for client in client_list:
                client_accum_gradient = client.train(global_round_idx, group_round_idx, w_group)
                client_to_gradient_dict[client.client_idx] = client_accum_gradient

            # Calculate sampling probability for each client
            gradient_sum_all_clients = 0
            for client_idx, gradient in client_to_gradient_dict.items():
                gradient_sum_all_clients += gradient
            client_to_probability_dict = {}
            for client_idx, gradient in client_to_gradient_dict.items():
                client_to_probability_dict[client_idx] = gradient / gradient_sum_all_clients

            # Dynamically sample clients
            client_idx_list = []
            probability_list = []
            for client_idx, probability in client_to_probability_dict.items():
                client_idx_list.append(client_idx)
                probability_list.append(probability)
            # num_clients = math.ceil(pow(0.9, global_round_idx) * len(client_list))
            num_clients = math.ceil(pow(sample_base_num, global_round_idx) * len(client_list))
            sampled_client_indexes = np.random.choice(client_idx_list, size=num_clients, replace=False, p=probability_list)
            logging.info("number of sampled clients for edge aggregate: {}".format(len(sampled_client_indexes)))
            for sampled_client_idx in sampled_client_indexes:
                w_local_list = self.client_dict[sampled_client_idx].send_weight()
                for i in range(len(w_local_list)):
                    quantized_w_list = OrderedDict()
                    for layer, w in w_local_list[i][1].items():
                        w_seed = w[2]
                        torch.manual_seed(w_seed)
                        dither = torch.rand(w[0].shape) - 0.5
                        quantized_w_list[layer] = torch.mul(w[0], w[1]) - dither
                    w_local_list[i] = (w_local_list[i][0], quantized_w_list)
                    # print(w_local_list)
                for global_epoch, w in w_local_list:
                        if not global_epoch in w_locals_dict: w_locals_dict[global_epoch] = []
                        w_locals_dict[global_epoch].append((client.get_sample_number(), w))

            # aggregate local weights
            for global_epoch in sorted(w_locals_dict.keys()):
                w_locals = w_locals_dict[global_epoch]
                w_group_list.append((global_epoch, self._aggregate(w_group, w_locals)))

            # update the group weight
            w_group = w_group_list[-1][1]
        return w_group_list

    def _aggregate(self, w_group, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = w_group[k] + local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params
