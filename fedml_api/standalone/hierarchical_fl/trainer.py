import logging
import numpy as np
import copy
import torch

from fedml_api.standalone.hierarchical_fl.group import Group
from fedml_api.standalone.hierarchical_fl.client import Client
from fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI

class Trainer(FedAvgAPI):

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict):
        logging.info("############setup_clients (START)#############")
        if self.args.group_method == 'random':
            if self.args.writers == []:
                self.group_indexes = np.random.randint(0, self.args.group_num, self.args.client_num_in_total)
            else:
                self.group_indexes = [0,0,0,1,1,1]
            group_to_client_indexes = {}
            for client_idx, group_idx in enumerate(self.group_indexes):
                if not group_idx in group_to_client_indexes:
                    group_to_client_indexes[group_idx] = []
                group_to_client_indexes[group_idx].append(client_idx)
        else:
            raise Exception(self.args.group_method)

        self.group_dict = {}
        for group_idx, client_indexes in group_to_client_indexes.items():
            self.group_dict[group_idx] = Group(group_idx, client_indexes, train_data_local_dict, test_data_local_dict,
                                               train_data_local_num_dict, self.args, self.device, self.model_trainer)

        # maintain a dummy client to be used in FedAvgTrainer::local_test_on_all_clients()
        self.client_list = [Client(client_idx, train_data_local_dict[0], test_data_local_dict[0],
                       train_data_local_num_dict[0], self.args, self.device, self.model_trainer)]
        logging.info("############setup_clients (END)#############")
    
    def _setup_clients_personalize(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict):
        logging.info("############setup_clients (START)#############")
        if self.args.group_method == 'random':
            if self.args.writers == []:
                self.group_indexes = np.random.randint(0, self.args.group_num, self.args.client_num_in_total)
            else:
                self.group_indexes = [0, 0, 0, 1, 1, 1]
            group_to_client_indexes = {}
            for client_idx, group_idx in enumerate(self.group_indexes):
                if not group_idx in group_to_client_indexes:
                    group_to_client_indexes[group_idx] = []
                group_to_client_indexes[group_idx].append(client_idx)
        else:
            raise Exception(self.args.group_method)

        self.group_dict = {}
        for group_idx, client_indexes in group_to_client_indexes.items():
            self.group_dict[group_idx] = Group(group_idx, client_indexes, train_data_local_dict, test_data_local_dict,
                                               train_data_local_num_dict, self.args, self.device, self.model_trainer)

        for client_idx in range(self.args.client_num_in_total):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, self.model_trainer)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def client_sampling(self, global_round_idx, client_num_in_total, client_num_per_round):
        sampled_client_indexes = super()._client_sampling(global_round_idx, client_num_in_total, client_num_per_round)
        group_to_client_indexes = {}
        for client_idx in sampled_client_indexes:
            group_idx = self.group_indexes[client_idx]
            if not group_idx in group_to_client_indexes:
                group_to_client_indexes[group_idx] = []
            group_to_client_indexes[group_idx].append(client_idx)
        logging.info("client_indexes of each group = {}".format(group_to_client_indexes))
        return group_to_client_indexes

    def train(self, personalize=False, communication=False, quantize_num=128, pow_base=0.9, dithered=False):
        w_global = self.model_trainer.model.state_dict()
        if communication:
            group_to_client_indexes = self.client_sampling(0, self.args.client_num_in_total,
                                                  self.args.client_num_per_round)
        for global_round_idx in range(self.args.global_comm_round):
            logging.info("################Global Communication Round : {}".format(global_round_idx))
            if communication is False:
                group_to_client_indexes = self.client_sampling(global_round_idx, self.args.client_num_in_total,
                                                  self.args.client_num_per_round)

            # train each group
            w_groups_dict = {}
            for group_idx in sorted(group_to_client_indexes.keys()):
                sampled_client_indexes = group_to_client_indexes[group_idx]

                ##################################
                if personalize:
                    for idx in sampled_client_indexes:
                        self.client_list[idx].sampled = True
                ##################################

                group = self.group_dict[group_idx]
                w_group_list = group.train(global_round_idx, w_global, sampled_client_indexes, personalize, communication, quantize_num, pow_base, dithered)
                #self.group_dict[group_idx] = copy.deepcopy(group)
                for global_epoch, w in w_group_list:
                    if not global_epoch in w_groups_dict: w_groups_dict[global_epoch] = []
                    w_groups_dict[global_epoch].append((group.get_sample_number(sampled_client_indexes), w))
                ####group_weight_dict[group_idx] = w_group_list[-1][1]

            ###################################
            # update self.client_list
            '''if personalize:
                for group_idx in sorted(group_to_client_indexes.keys()):
                    sampled_client_indexes = group_to_client_indexes[group_idx]
                    group = self.group_dict[group_idx]
                    print("group idx", group_idx, sampled_client_indexes)
                    sampled_client_list = [group.client_dict[client_idx] for client_idx in sampled_client_indexes]
                    for client in sampled_client_list:
                        m = client.local_test(False)
                        #print(client.client_idx, m['test_correct'], m['test_total'])
                        self.client_list[client.client_idx].model_trainer.model.load_state_dict(client.model_trainer.model.state_dict())'''
            #self._local_test_on_all_clients(0)
            ###################################  
            
            # aggregate group weights into the global weight
            for global_epoch in sorted(w_groups_dict.keys()):
                w_groups = w_groups_dict[global_epoch]
                w_global = self._aggregate(w_groups)

                # evaluate performance
                if (global_epoch+1) % self.args.epochs == 0 or \
                    global_epoch == self.args.global_comm_round*self.args.group_comm_round*self.args.epochs-1:
                    if not personalize:
                        self.model_trainer.model.load_state_dict(w_global)
                    #############################################
                    elif self.args.writers == []:
                        for client_idx in range(self.args.client_num_in_total):
                            if self.client_list[client_idx].sampled == False:
                                self.client_list[client_idx].model_trainer.model.load_state_dict(w_global)
                    #############################################
                    if personalize:
                        for group_idx in sorted(group_to_client_indexes.keys()):
                            sampled_client_indexes = group_to_client_indexes[group_idx]
                            group = self.group_dict[group_idx]
                            sampled_client_list = [group.client_dict[client_idx] for client_idx in sampled_client_indexes]
                            for client in sampled_client_list:
                                #model_path = "./client_"+str(client.client_idx)+".pt"
                                #torch.save(client.model_trainer.model.state_dict(), model_path)
                                ###m = client.local_test(False)
                                self.client_list[client.client_idx] = copy.deepcopy(client)
                                #self.client_list[client.client_idx].model_trainer.model.load_state_dict(client.model_trainer.model.state_dict())
                    else:
                        model_path = "./global.pt"
                        torch.save(w_global, model_path)
                    self._local_test_on_all_clients(global_epoch)
