import torch

class Quantizer:
    def __init__(self, tensor, num_bits=8, symmetric=True):
        self.num_bits = num_bits
        self.tensor = tensor
        self.symmetric = symmetric

    def get_min_max(self):
        self.abs_min = torch.min(self.tensor)
        self.abs_max = torch.max(self.tensor)
        self.quantile_min = torch.quantile(self.tensor, 0.01)
        self.quantile_max = torch.quantile(self.tensor, 0.99)
        if self.num_bits == 8:
            if self.symmetric:
                self.qmin = -127
                self.qmax = 127
            else:
                self.qmin = -128
                self.qmax = 127
        if self.num_bits == 16:
            if self.symmetric:
                self.qmin = -32767
                self.qmax = 32767
            else:
                self.qmin = -32768
                self.qmax = 32767
    
    def calculate_scale_offset(self):
        # q = f / scale + offset
        self.scale = (self.quantile_max - self.quantile_min) / (self.qmax - self.qmin)
        self.offset = self.qmin - self.quantile_min / self.scale

    def quantize(self):
        self.get_min_max()
        self.calculate_scale_offset()
        self.quantized_tensor = torch.clamp(self.tensor, min=self.quantile_min, max=self.quantile_max)
        self.quantized_tensor = torch.mul(self.quantized_tensor, 1 / self.scale)
        self.quantized_tensor = torch.add(self.quantized_tensor, self.offset)
        self.quantized_tensor = torch.round(self.quantized_tensor)
        return self.quantized_tensor

