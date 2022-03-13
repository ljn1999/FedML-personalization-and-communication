import copy
import torch
import numpy
from torch import nn
from numpy import linalg

from fedml_api.standalone.fedavg.client import Client
from fedml_api.standalone.hierarchical_fl.quantizer import Quantizer

class Client(Client):

    def train(self, global_round_idx, group_round_idx, w, personalize=False, communication=False):
        w_group = copy.deepcopy(w)
        model = self.model_trainer.model
        if personalize:
            '''if group_round_idx == 0 and global_round_idx == 0:
                w_client = w
                self.weights = copy.deepcopy(w)'''
            if self.weights == {}: # if uninitialized
                w_client = copy.deepcopy(w)
                self.weights = copy.deepcopy(w)
            else:
                w_client = copy.deepcopy(self.weights)
                for k in list(w_client.keys())[0:4]: # for the global layer (conv2d_1.weight and conv2d_1.bias)
                                                    # and group layer (conv2d_2.weight and conv2d_2.bias)
                    w_client[k] = copy.deepcopy(w[k]) 
        else:
            w_client = w
        
        model.load_state_dict(w_client)
        model.to(self.device)
        '''train_local_metrics = self.local_test(False)
        print("client idx:", self.client_idx, "w_client acc:", train_local_metrics['test_correct']/train_local_metrics['test_total'])'''

        #criterion = nn.CrossEntropyLoss().to(self.device)
        #print("model name:", model.__class__.__name__)
        
        if model.__class__.__name__ == "CNN_DropOut_Binary":
            criterion = nn.BCEWithLogitsLoss().to(self.device)
        else:
            criterion = nn.CrossEntropyLoss().to(self.device)

        if self.args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model_trainer.model.parameters(), lr=self.args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model_trainer.model.parameters()), lr=self.args.lr,
                                              weight_decay=self.args.wd, amsgrad=True)

        w_list = []
        accum_gradient = 0
        for epoch in range(self.args.epochs):
            '''if epoch == 0:
               train_local_metrics = self.local_test(False)
               print("client idx:", self.client_idx, "right after aggr acc:", train_local_metrics['test_correct']/train_local_metrics['test_total'])'''
            for x, labels in self.local_training_data:
                x, labels = x.to(self.device), labels.to(self.device)
                self.model_trainer.model.zero_grad()
                log_probs = self.model_trainer.model(x)
                if model.__class__.__name__ == "CNN_DropOut_Binary":
                    loss = criterion(log_probs, labels.unsqueeze(1).type(torch.FloatTensor).cuda())
                else:
                    loss = criterion(log_probs, labels)
                loss.backward()
                if communication:
                    for param in self.model_trainer.model.parameters():
                        accum_gradient += linalg.norm(param.grad.cpu())
                optimizer.step()
            global_epoch = global_round_idx*self.args.group_comm_round*self.args.epochs + \
                            group_round_idx*self.args.epochs + epoch
            if global_epoch % self.args.frequency_of_the_test == 0 or epoch == self.args.epochs-1:
                if communication:
                    w_orig = copy.deepcopy(self.model_trainer.model.state_dict())
                    w_quantized = copy.deepcopy(self.model_trainer.model.state_dict())
                    # quantize weight
                    for layer, weight in w_orig.items():
                        quantizer = Quantizer(torch.sub(weight.cpu(), w_group[layer]))
                        # hardcode s = 256 for testing for now
                        w_norm, w_L = quantizer.quantize(256)
                        w_quantized[layer] = (w_L, w_norm / 256)
                    w_list.append((global_epoch, w_quantized))
                else:
                    w_list.append((global_epoch, copy.deepcopy(self.model_trainer.model.state_dict())))
            self.weights = copy.deepcopy(self.model_trainer.model.state_dict())
        # train acc
        train_local_metrics = self.local_test(False)
        #print("client idx:", self.client_idx, "epoch training acc:", train_local_metrics['test_correct'], train_local_metrics['test_total'])
        #print("client idx:", self.client_idx, "training loss:", train_local_metrics['test_loss'], train_local_metrics['test_total'])
        # test acc
        test_local_metrics = self.local_test(True)
        #print("client idx:", self.client_idx, "test acc:", test_local_metrics['test_correct'], test_local_metrics['test_total'])
        #print("client idx:", self.client_idx, "test loss:", test_local_metrics['test_loss'], test_local_metrics['test_total'])
        
        self.client_weight_list = w_list
        if communication:
            weight_difference = 0
            for layer, weight in self.weights.items():
                weight_difference += linalg.norm(torch.sub(weight.cpu(), w_group[layer]))
            return accum_gradient, weight_difference
        else:
            return w_list
    
    def send_weight(self):
        return self.client_weight_list

