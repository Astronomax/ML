from collections import OrderedDict

class MLP(nn.Module):
    """
        Simple linear network — multi-layer perceptron
    """
    def __init__(self, input_size, num_layers, hidden_sizes, output_size, activations, dropouts):
        super(MLP, self).__init__()
        
        if not isinstance(hidden_sizes, list):
            hidden_sizes = [hidden_sizes] * num_layers
        assert len(hidden_sizes) == num_layers, f'provide {num_layers} hidden_sizes or just one for all layers'
        
        if not isinstance(activations, list):
            activations = [activations] * num_layers
        assert len(activations) == num_layers, f'provide {num_layers} activation functions or just one for all layers'
        
        if not isinstance(dropouts, list):
            dropouts = [dropouts] * num_layers
        assert len(dropouts) == num_layers, f'provide {num_layers} dropout values or just one for all layers'
        
        flat = ('flat', nn.Flatten())
        in_to_hid = ('in2hid', nn.Linear(input_size, hidden_sizes[0]))
        
        hid_ = [[
            (f'act_{i+1}', activations[i]),
            (f'drop_{i+1}', nn.Dropout(dropouts[i])),
            (f'hid_{i+1}', nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        ] for i in range(num_layers-1)]
        
        hid = []
        for el in hid_:
            hid.extend(el)
        
        head = [
            (f'act_{num_layers}', activations[-1]),
            (f'drop_{num_layers}', nn.Dropout(dropouts[-1])),
            ('hid2out', nn.Linear(hidden_sizes[-1], output_size)),
            ('log-softmax', nn.LogSoftmax(dim=-1))
        ]
        
        self.net = [flat, in_to_hid, *hid, *head]
        self.net = nn.Sequential(OrderedDict(self.net))
    
    def forward(self, imgs):
        return self.net(imgs)