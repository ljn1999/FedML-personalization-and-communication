import torch
from torch import linalg as LA
import torch.nn as nn
import numpy as np

class Quantizer:
    def __init__(self, tensor):
        self.tensor = tensor
        self.norm = -1
    
    def quantize(self, s):
        # calculate the norm
        self.sq_norm = LA.norm(self.tensor)
        # get the sign vector
        self.signs = torch.sign(self.tensor)
        # L = floor(|vi| * s / norm)
        self.L = torch.mul(torch.abs(self.tensor), (s / self.sq_norm))
        self.L = torch.round(self.L)
        # # return norm and signed L: O(1 + n)
        self.L = torch.mul(self.L, self.signs)
        # add parameter to decide which integer type later
        self.L = self.L.int()
        return self.sq_norm, self.L