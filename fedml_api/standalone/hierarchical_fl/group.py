import logging
import math
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

    def train(self, global_round_idx, w, client_indexes):
        client_list = [self.client_dict[client_idx] for client_idx in client_indexes]
        w_group = w
        w_group_list = []
        for group_round_idx in range(self.args.group_comm_round):
            logging.info("Group ID : {} / Group Communication Round : {}".format(self.idx, group_round_idx))
            w_locals_dict = {}

            # train each client
            client_to_gradient_dict = {}
            for client in client_list:
                w_local_list, client_accum_gradient = client.train(global_round_idx, group_round_idx, w_group)
                client_to_gradient_dict[client.client_idx] = client_accum_gradient
                for global_epoch, w in w_local_list:
                    if not global_epoch in w_locals_dict: w_locals_dict[global_epoch] = []
                    w_locals_dict[global_epoch].append((client.client_idx, client.get_sample_number(), w))
            client_to_gradient_dict = dict(sorted(client_to_gradient_dict.items(), key=lambda item: item[1], reverse=True))
            # Dynamically sample clients
            num_clients = math.ceil(pow(0.9, global_round_idx) * len(client_list))
            sampled_client_indexes = []
            count = 0
            for client_idx, gradient in client_to_gradient_dict.items():
                if count == num_clients:
                    break
                count += 1
                sampled_client_indexes.append(client_idx)
            sampled_w_locals_dict = {}
            for global_epoch, c_list in w_locals_dict.items():
                for tup in c_list:
                    if not tup[0] in sampled_client_indexes:
                        break
                    if not global_epoch in sampled_w_locals_dict: sampled_w_locals_dict[global_epoch] = []
                    sampled_w_locals_dict[global_epoch].append((tup[1], tup[2]))
            # aggregate local weights
            for global_epoch in sorted(sampled_w_locals_dict.keys()):
                w_locals = sampled_w_locals_dict[global_epoch]
                w_group_list.append((global_epoch, self._aggregate(w_locals)))

            # update the group weight
            w_group = w_group_list[-1][1]
        return w_group_list
