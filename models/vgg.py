import torch
import torch.nn as nn
import sys
from torchsummary import summary as summary_

class VGG(nn.Module):
    def __init__(self, num_layers, num_classes=2, init_weights=True):
        super(VGG, self).__init__()
        # input image size (N, 3, 224, 224)
        # after maxpooling layer, h and w are devided by 2 : 224->112->56->28->14->7
        self.in_channels = 3
        # there are out_channels and M(maxpool) in self.vgg_cfg 
        if num_layers==11:
            self.vgg_cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        elif num_layers==13:
            self.vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        elif num_layers==16:
            self.vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        elif num_layers==19:
            self.vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        else: 
            print("unavailable number of layers")
            sys.exit()

        self.conv_layers = self._make_layers(self.vgg_cfg)
        # fc layers part : adaptiveaveragepooling->FC->ReLU->Dropout->FC->ReLU->Dropout->FC (-> softmax)
        self.adaptive_avgpooling = nn.AdaptiveAvgPool2d(7)
        self.fc_layers = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

        # placeholder for the gradients
        self.gradients = None

        if init_weights:
            self._initialize_weights()

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(v), 
                            nn.ReLU()]
                in_channels = v
        return nn.Sequential(*layers)
    
    def forward(self, x):
        output = self.conv_layers(x)
        
        h = x.register_hook(self.activations_hook)
        
        output = self.adaptive_avgpooling(output)
        output = output.view(-1, 512*7*7)        
        output = self.fc_layers(output)
        return output
    
    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.conv_layers(x)

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


if __name__ == "__main__":
    vgg = VGG(16)
    x = torch.zeros((1, 3, 224, 224))
    
    pred = vgg(x)
    print(pred)