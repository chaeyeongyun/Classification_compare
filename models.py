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
        ResidualBlock.expansion=1 
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels*ResidualBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels*ResidualBlock.expansion)
        )
        # the case that the number of input channels is not equal with the number of output channels
        # in other cases, the result of self.shortcut is x
        if stride != 1 or in_channels != out_channels * ResidualBlock.expansion:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels, out_channels*ResidualBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*ResidualBlock.expansion)
            )
        else:
            self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_layers(x) + self.shortcut(x)
        x = self.relu(x)
        return x

        
class BottleNeck(nn.Module):
    '''
    in 50, 101, 152-layer resnet
    '''
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()
        # when the number of out channels is increased : in bottleneck architecture, use zeropadding, in basic block, use projection shortcut(F(x)+W_s * x)
        BottleNeck.expansion = 4 # Table 1: 128*4=512, 256*4=1024, 512*4=2048
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels*BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels*BottleNeck.expansion)
        )
        # the case that the number of input channels is not equal with the number of output channels
        # in other cases, the result of self.shortcut is x
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*BottleNeck.expansion)
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
    def __init__(self, num_layers, num_classes=2, init_weights=True):
        super(Resnet, self).__init__()
        if num_layers == 18:
            num_blocks = [2, 2, 2, 2]
            block_class = ResidualBlock
        elif num_layers == 34:
            num_blocks = [3, 4, 6, 3]
            block_class = ResidualBlock
        elif num_layers == 50:
            num_blocks = [3, 4, 6, 3]
            block_class = BottleNeck
        elif num_layers == 101:
            num_blocks = [3, 4, 23, 3]
            block_class = BottleNeck
        elif num_layers == 152:
            num_blocks = [3, 8, 36, 3]
            block_class = BottleNeck
        else: 
            print(" It's not appropriate number of layers ") 
            sys.exit(0)
        
        self.in_channels = 64
        self.conv1_n_maxpool = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2_x = self._make_layers(block_class, num_blocks[0], 1, 64) # N, 64, 56, 56
        self.conv3_x = self._make_layers(block_class, num_blocks[1], 2, 128) # N, 128, 28, 28
        self.conv4_x = self._make_layers(block_class, num_blocks[2], 2, 256)
        self.conv5_x = self._make_layers(block_class, num_blocks[3], 2, 512)
        self.average_pool = nn.AdaptiveAvgPool2d((1,1)) # N, C, 1, 1
        self.fc_layer = nn.Linear(512*block_class.expansion, num_classes)
    
    def _make_layers(self, block_class, num_blocks, stride, out_channels):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers = layers + [block_class(self.in_channels, out_channels, stride)]
            self.in_channels = out_channels * block_class.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        output = self.conv1_n_maxpool(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.average_pool(output)
        print(output.size())
        output = output.view(output.size(0), -1)
        output = self.fc_layer(output)
        return output
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # He initialization
                if m.bias is not None: 
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01) # Fills the given 2-dimensional matrix with values drawn from a normal distribution parameterized by mean and std.
                nn.init.constant_(m.bias, 0)


# if __name__ == "__main__":
#     # resnet = Resnet(18)
#     # # resnet = Resnet(34)
#     # # resnet = Resnet(50)
#     # # resnet = Resnet(101)
#     # # resnet = Resnet(152)
#     # x = torch.zeros((1, 3, 224, 224))
#     # resnet(x)