import torch
import torch.nn as nn
from collections import OrderedDict

'''
hidden_dim=100 just works
'''
class SimpleModel(nn.Module):
    def __init__(self, hidden_dim: int = 100):
        super().__init__()
        # HERE YOUR CODE
        flat = ('flat', nn.Flatten())
        in_to_hid = ('in2hid', nn.Linear(784, hidden_dim))
        
        head = [
            (f'act', nn.Tanh()),
            (f'drop', nn.Dropout(0.2)),
            ('hid2out', nn.Linear(hidden_dim, 10)),
            ('log-softmax', nn.LogSoftmax(dim=-1))
        ]
        
        self.net = [flat, in_to_hid, *head]
        self.net = nn.Sequential(OrderedDict(self.net))
        # TILL THERE

    def forward(self, images):
        # HERE YOU CODE
        return self.net(images)
        # TILL THERE