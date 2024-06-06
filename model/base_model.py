import os
import torch
import torch.nn as nn


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(
            'cuda' if opt['gpu_ids'] is not None else 'cpu')
    
        self.begin_step = 0
        self.begin_epoch = 0
        

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def set_device(self, x):
        if isinstance(x, torch.nn.Module):
            return x.to(self.device)
        elif isinstance(x, torch.Tensor):
            return x.to(self.device)
        elif isinstance(x, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
        elif isinstance(x, list):
            return [self.set_device(v) for v in x]
        else:
            return x

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n
