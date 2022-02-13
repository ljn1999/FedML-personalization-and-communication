import torch
from torch import linalg as LA
import torch.nn as nn
import numpy as np

class Quantizer:
    def __init__(self, tensor, num_bits=8, symmetric=True):
        self.num_bits = num_bits
        self.tensor = tensor
        self.symmetric = symmetric
        self.norm = -1

    # def get_min_max(self):
    #     self.abs_min = torch.min(self.tensor)
    #     self.abs_max = torch.max(self.tensor)
    #     self.quantile_min = torch.quantile(self.tensor, 0.01)
    #     self.quantile_max = torch.quantile(self.tensor, 0.99)
    #     if self.num_bits == 8:
    #         if self.symmetric:
    #             self.qmin = -127
    #             self.qmax = 127
    #         else:
    #             self.qmin = -128
    #             self.qmax = 127
    #     if self.num_bits == 16:
    #         if self.symmetric:
    #             self.qmin = -32767
    #             self.qmax = 32767
    #         else:
    #             self.qmin = -32768
    #             self.qmax = 32767
    
    # def calculate_scale_offset(self):
    #     # q = f / scale + offset
    #     self.scale = (self.quantile_max - self.quantile_min) / (self.qmax - self.qmin)
    #     self.offset = self.qmin - self.quantile_min / self.scale

    # def quantize(self):
    #     self.get_min_max()
    #     self.calculate_scale_offset()
    #     self.quantized_tensor = torch.clamp(self.tensor, min=self.quantile_min, max=self.quantile_max)
    #     self.quantized_tensor = torch.mul(self.quantized_tensor, 1 / self.scale)
    #     self.quantized_tensor = torch.add(self.quantized_tensor, self.offset)
    #     self.quantized_tensor = torch.round(self.quantized_tensor)
    #     print(self.quantized_tensor.shape)
    #     return self.quantized_tensor
    
    def quantize2(self, s):
        # calculate the norm
        self.sq_norm = LA.norm(self.tensor)
        # get the sign vector
        self.signs = torch.sign(self.tensor)
        # L = floor(|vi| * s / norm)
        self.L = torch.mul(torch.abs(self.tensor), (s / self.sq_norm))
        self.L = torch.round(self.L)
        ############# trial with probability: too slow ########
        # self.L = torch.floor(self.L)
        # # trial: flatten all tensors to 1D then revert back
        # self.shape = self.tensor.shape
        # # print("!!!!original shape:") 
        # # print(self.shape)
        # self.diff = torch.flatten(torch.mul(torch.abs(self.tensor), (s / self.sq_norm)) - self.L)
        # self.L = torch.flatten(self.L)
        # floor = self.L.numpy()
        # ceil = floor + 1
        # value = np.stack((floor, ceil), axis=-1)
        # diff_f = (torch.flatten(self.diff)).numpy()
        # diff_c = 1 - diff_f
        # p = np.stack((diff_c, diff_f), axis=-1)
        # for i in range(p.shape[0]):
        #     # print(value[i])
        #     # print(p[i])
        #     self.L[i] = (np.random.choice(value[i], size=1, p=p[i]).astype(int)).item()
            # print(self.L[i])
        # for i in range(0, self.L.size(dim=0)):
        # #     # p: [probability to floor, probability to ceil]
        #     # print("loop start:", i)
        #     p = [self.diff[i].item(), (1 - self.diff[i].item())]
        #     floor_ceil = [(self.L[i].item() + 1), self.L[i].item()]
        # #     choice = p.multinomial(num_samples=1, replacement=True)
        #     choice = np.random.choice(floor_ceil, size=1, p=p)
        #     # self.L[i] = floor_ceil[choice]
        #     self.L[i] = choice[0]
        #     # print("loop finish:", i)
        # self.m = nn.Sequential(nn.Unflatten(0, self.shape))
        # self.L = self.m(self.L)
        # # print("!!!!!!!!!!!later shape:") 
        # # print(self.L.shape)
        # # return norm and signed L: O(1 + n)
        self.L = torch.mul(self.L, self.signs)
        # add parameter to decide which integer type later
        self.L = self.L.int()
        return self.sq_norm, self.L

