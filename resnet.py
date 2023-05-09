import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    '''
    This class should be implemented similarly to the BasicBlock from the pytorch implementation.
    
    Since in resnet18 all blocks have only two parameters that are responsible for the number of channels,
    we will also use a simplified notation.

    The first convolutional layer has the dimension (in_channels, out_channels), 
    the second convolutional layer has the dimension (out_channels, out_channels).

    You are required to implement the correct forward() and __init__() methods for this block.
    
    Remember to use batch normalizations and activation functions.
    Shorcut will require you to understand what projection convolution is.

    In general, it is recommended to refer to the original article, the pytorch implementation and
    other sources of information to successfully assemble this block.

    Hint! You can use nn.Identity() to implement shorcut.
    '''

    def __init__(self, in_channels, out_channels):
        '''
        The block must have the following fields:
            *self.shorcut
            *self.activation
            *self.conv1
            *self.conv2
            *self.bn1
            *self.bn2

        Hint! Don't forget the bias, padding, and stride parameters for convolutional layers.
        '''
        super().__init__()
        stride = (2, 2) if in_channels != out_channels else (1, 1)
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
        
        self.activation = nn.ReLU(inplace=True)
        
        self.hid = nn.Sequential(OrderedDict([
            ('hid_1', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)),
            ('bn_1', nn.BatchNorm2d(out_channels)),
            ('act_1', self.activation),
            ('hid_2', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn_2', nn.BatchNorm2d(out_channels))
        ]))


    def forward(self, x):
        residual = self.shortcut(x)
        return self.activation(self.hid(x) + residual)


class ResNetLayer(nn.Module):
    '''
    This class should be implemented similarly to layer from the pytorch implementation.
    
    To implement the layer, you will need to create two ResidualBlocks inside.
    Determining the appropriate dimensions is up to you.
    '''
    
    def __init__(self, in_channels, out_channels):
        '''
        The layer must have the following field declared:
            *self.blocks
        '''
        super().__init__()
        
        self.blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(out_channels, out_channels)
        )
        
    def forward(self, x):
        '''
        Note that blocks must be packed to make forward work in its original form.
        ''' 
        x = self.blocks(x)
        return x


class ResNet18(nn.Module):
    '''
    Finally, this class should consist of three main components:
      1. Four preparatory layers
      2. A set of internal ResNetLayers
      3. Final classifier

    Hint! In order for the network to process images from CIFAR10, you should replace the parameters
          of the first convolutional layer on kernel_size=(3, 3), stride=(1, 1) and padding=(1, 1).
    '''

    def __init__(self, in_channels=3, n_classes=10):
        '''
        The class must have the following fields declared:
            *self.conv1
            *self.bn1
            *self.activation
            *self.maxpool
            *self.layers
            *self.avgpool
            *self.flatten
            *self.fc

         A different grouping of parameters is allowed that does not violate the idea of the network architecture.
        '''

        super().__init__()
        
        self.conv_1 = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        
        self.net = nn.Sequential(OrderedDict([
            ('conv_1', self.conv_1),
            ('bn_1', nn.BatchNorm2d(64)),
            ('act_1', nn.ReLU(inplace=True)),
            ('maxpool', nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))),
            ('residual_1', ResNetLayer(64, 64)),
            ('residual_2', ResNetLayer(64, 128)),
            ('residual_3', ResNetLayer(128, 256)),
            ('residual_4', ResNetLayer(256, 512)),
            ('avgpool', nn.AdaptiveAvgPool2d(output_size=(1, 1))),
            ('flat', nn.Flatten()),
            ('fc', nn.Linear(512, n_classes))
        ]))

    def forward(self, x):
        return self.net(x)