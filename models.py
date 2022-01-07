import torch
import torch.nn as nn
import sys

class ResidualBlock(nn.Module):
    '''
    F(x)+x
    '''
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # when the number of out channels is increased : in bottleneck architecture, use zeropadding, in basic block, use projection shortcut(F(x)+W_s * x)
        self.expansion=1 
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels*self.expansion)
        )
        # the case that the number of input channels is not equal with the number of output channels
        # in other cases, the result of self.shortcut is x
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*self.expansion)
            )
        else:
            self.shortcut = nn.Sequential()
    
    def forward(self, x):
        x = self.conv_layers(x) + self.shortcut(x)
        return nn.ReLU(x)

        
class BottleNeck(nn.Module):
    '''
    in 50, 101, 152-layer resnet
    '''
    def __init__(self, in_channels, out_channels, stride=1):
        self.expansoin = 4
        super(BottleNeck, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels*self.expansion)
        )
        # the case that the number of input channels is not equal with the number of output channels
        # in other cases, the result of self.shortcut is x
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*self.expansion)
            )
        else:
            self.shortcut = nn.Sequential()
        
        def forward(self, x):
            x = self.conv_layers(x) + self.shortcut(x)
            return nn.ReLU(x)

class Resnet(nn.Module):
    '''
    num_layers : the number of layers of ResNet ( i.e. 18, 34, 50, 101, 152 )
    '''
    def __init__(self, num_layers):
        super(Resnet, self).__init__()
        if num_layers == 18:
            num_blocks = [2, 2, 2, 2]
        elif num_layers == 34:
            num_blocks = [2, 2, 2, 2]
        elif num_layers == 50:
            num_blocks = [2, 2, 2, 2]
        elif num_layers == 101:
            num_blocks = [2, 2, 2, 2]
        elif num_layers == 152:
            num_blocks = [2, 2, 2, 2]
        else: 
            print(" It's not appropriate number of layers ") 
            sys.exit(0)
        
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False), 
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        )
        self.conv3_x
        self.conv4_x
        self.conv5_x
        self.average_pool
        self.fc_layer
        self.softmax
    
    def _make_layers(self, num_blocks):
        