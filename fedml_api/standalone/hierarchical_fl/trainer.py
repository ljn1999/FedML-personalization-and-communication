import logging
import numpy as np
import math

from fedml_api.standalone.hierarchical_fl.group import Group
from fedml_api.standalone.hierarchical_fl.client import Client
from fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI

class Trainer(FedAvgAPI):

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict):
        logging.info("############setup_clients (START)#############")
        # Assign a group index to each client
        # Create a dict group_to_client_indexes whose key is the group index and the value is a list of client indexes in that group
        if self.args.group_method == 'random':
            # !! Modify total number of clients here
            self.group_indexes = np.random.randint(0, self.args.group_num, 30)
            group_to_client_indexes = {}
            for client_idx, group_idx in enumerate(self.group_indexes):
                if not group_idx in group_to_client_indexes:
                    group_to_client_indexes[group_idx] = []
                group_to_client_indexes[group_idx].append(client_idx)
        else:
            raise Exception(self.args.group_method)

        # for each group/cluster: make a Group object with this information:
        # group idx; client_indexes: the indices of clients in this group
        # train data & test data
        self.group_dict = {}
        for group_idx, client_indexes in group_to_client_indexes.items():
            self.group_dict[group_idx] = Group(group_idx, client_indexes, train_data_local_dict, test_data_local_dict,
                                               train_data_local_num_dict, self.args, self.device, self.model_trainer)

        # logging.info("train_data_local_dict[0]:{}".format(train_data_local_dict[0]))
        # logging.info("test_data_local_dict[0]:{}".format(test_data_local_dict[0]))
        # logging.info("train_data_local_num_dict:{}".format(train_data_local_num_dict))
        # maintain a dummy client to be used in FedAvgTrainer::local_test_on_all_clients()
        self.client_list = [Client(client_idx, train_data_local_dict[0], test_data_local_dict[0],
                       train_data_local_num_dict[0], self.args, self.device, self.model_trainer)]
        logging.info("############setup_clients (END)#############")

    def client_sampling(self, global_round_idx, client_num_in_total, client_num_per_round):
        # !! Modify total number of clients here
        sampled_client_indexes = super()._client_sampling(global_round_idx, 30, 30)
        logging.info("Total number of clients : {}".format(len(sampled_client_indexes)))
        group_to_client_indexes = {}
        for client_idx in sampled_client_indexes:
            group_idx = self.group_indexes[client_idx]
            if not group_idx in group_to_client_indexes:
                group_to_client_indexes[group_idx] = []
            group_to_client_indexes[group_idx].append(client_idx)
        return group_to_client_indexes
    
    # Dynamic sampling test
    def client_sampling_dynamic(self, global_round_idx, client_num_in_total, client_num_per_round):
        # For global sampling, we ignore the argument client_num_per_round passed in by the user
        logging.info("ENTERING DYNAMIC SAMPLING")
        sampled_client_indexes = super()._client_sampling(global_round_idx, 20, 20)
        group_to_client_indexes = {}
        for client_idx in sampled_client_indexes:
            group_idx = self.group_indexes[client_idx]
            if not group_idx in group_to_client_indexes:
                group_to_client_indexes[group_idx] = []
            group_to_client_indexes[group_idx].append(client_idx)
        # Apply dynamic sampling to each group
        for group_idx, client_indexes in group_to_client_indexes.items():
            num_clients = math.ceil(pow(0.9, global_round_idx) * len(client_indexes))
            np.random.seed(global_round_idx)  # make sure for each comparison, we are selecting the same clients each round
            group_to_client_indexes[group_idx] = np.random.choice(client_indexes, num_clients, replace=False)
        logging.info("SANITY DYNAMIC SAMPLING: client_indexes of each group = {}".format(group_to_client_indexes))
        return group_to_client_indexes

    def train(self):
        w_global = self.model_trainer.model.state_dict()
        group_to_client_indexes = self.client_sampling(0, self.args.client_num_in_total,
                                                  self.args.client_num_per_round)
        for global_round_idx in range(self.args.global_comm_round):
            logging.info("################Global Communication Round : {}".format(global_round_idx))
            # group_to_client_indexes = self.client_sampling(global_round_idx, self.args.client_num_in_total,
                                                #   self.args.client_num_per_round)

            # train each group
            w_groups_dict = {}
            for group_idx in sorted(group_to_client_indexes.keys()):
                sampled_client_indexes = group_to_client_indexes[group_idx]
                group = self.group_dict[group_idx]
                w_group_list = group.train(global_round_idx, w_global, sampled_client_indexes)
                for global_epoch, w in w_group_list:
                    if not global_epoch in w_groups_dict: w_groups_dict[global_epoch] = []
                    w_groups_dict[global_epoch].append((group.get_sample_number(sampled_client_indexes), w))

            # aggregate group weights into the global weight
            for global_epoch in sorted(w_groups_dict.keys()):
                w_groups = w_groups_dict[global_epoch]
                w_global = self._aggregate(w_groups)

                # evaluate performance
                if global_epoch % self.args.frequency_of_the_test == 0 or \
                    global_epoch == self.args.global_comm_round*self.args.group_comm_round*self.args.epochs-1:
                    self.model_trainer.model.load_state_dict(w_global)
                    self._local_test_on_all_clients(global_epoch)
