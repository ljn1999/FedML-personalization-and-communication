import torch
from torch import linalg as LA
import torch.nn as nn
import numpy as np
import time

class Quantizer:
    def __init__(self, tensor):
        self.tensor = tensor
        self.norm = -1
    
    def quantize(self, s):
        # calculate the norm
        self.sq_norm = LA.norm(self.tensor)
        # get the sign vector
        self.signs = torch.sign(self.tensor)
        # add dithered value
        self.seed = int(time.time())
        torch.manual_seed(self.seed)
        self.dither = torch.rand(self.tensor.shape) - 0.5
        # L = floor(|vi| * s / norm)
        self.L = torch.mul(torch.abs(self.tensor + self.dither), (s / self.sq_norm))
        self.L = torch.round(self.L)
        # # return norm and signed L: O(1 + n)
        self.L = torch.mul(self.L, self.signs)
        # add parameter to decide which integer type later
        self.L = self.L.int()
        return self.sq_norm, self.L, self.seed
