import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    '''
    F(x)+x
    '''
    def __init__(self, in_channels, out_chaennels, stride=1):
        super(ResidualBlock, self).__init__()
        # when the number of out channels is increased : in bottleneck architecture, use zeropadding, in basic block, use projection shortcut(F(x)+W_s * x)
        self.expansion=1 
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_chaennels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_chaennels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_chaennels*self.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_chaennels*self.expansion)
        )
        # the case that the number of input channels is not equal with the number of output channels
        # in other cases, the result of self.shortcut is x
        if stride != 1 or in_channels != out_chaennels * self.expansion:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels, out_chaennels*self.expansion, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.shortcut = nn.Sequential()
    def forward(self, x):
        x = self.conv_layers(x) + self.shortcut(x)
        return nn.ReLU(x)

        
class BottleNeck(nn.Module):
    def __init__(self):
        super(BottleNeck, self).__init__()

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
