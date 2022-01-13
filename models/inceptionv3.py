import torch
import torch.nn as nn
import sys

class InceptionA(nn.Module):
    # 논문과 구현 코드가 조금 다름...(사실 많이 ;;)
    def __init__(self, in_channels, pool_features, stride=1):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d()
        )
        self.branch5x5_1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=1),
            nn.BatchNorm2d()
        )
        self.branch5x5_2 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d()
        )

        self.branch3x3_1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d()
        )
        self.branch3x3_2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d()
        )
        self.branch3x3_3 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d()
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_features, kernel_size=1),
            nn.BatchNorm2d()
        )
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
        branchpool = self.branch_pool(x)
        outputs = [branch1x1, branch5x5, branch3x3, branchpool]
        output = torch.cat(outputs, 1)
        return output

class InceptionB(nn.Module):
    def __init__(self, in_channels, stride=1):
        super(InceptionB, self).__init__()
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 384, kernel_size=3, stride=2),
            nn.BatchNorm2d()
        )

        self.branch1x1_3x3_1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d()
        )
        self.branch1x1_3x3_2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d()
        )
        self.branch1x1_3x3_3 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=2),
            nn.BatchNorm2d()
        )
        self.branchpool = nn.AvgPool2d(kernel_size=3, stride=2)
    
    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch1x1_3x3 = self.branch1x1_3x3_1(x)
        branch1x1_3x3 = self.branch1x1_3x3_2(branch1x1_3x3)
        branch1x1_3x3 = self.branch1x1_3x3_3(branch1x1_3x3)
        branchpool = self.branchpool(x)
        outputs = [branch3x3, branch1x1_3x3, branchpool]
        output = torch.cat(outputs, 1)
        return output

class InceptionC(nn.Module):
    def __init__(self, in_channels, out_channels_7x7, stride=1):
        super(InceptionC, self).__init__()
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=1),
            nn.BatchNorm2d()
        )
        
        self.branch7x7_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_7x7, kernel_size=1),
            nn.BatchNorm2d()
        )
        self.branch7x7_2 = nn.Sequential(
            nn.Conv2d(out_channels_7x7, out_channels_7x7, kernel_size=(1, 7), padding=(0,3)),
            nn.BatchNorm2d()
        )
        self.branch7x7_3 = nn.Sequential(
            nn.Conv2d(out_channels_7x7, 192, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d()
        )

        self.branchdouble7x7_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_7x7, kernel_size=1),
            nn.BatchNorm2d()
        )
        self.branchdouble7x7_2 = nn.Sequential(
            nn.Conv2d(out_channels_7x7, out_channels_7x7, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d()
        )
        self.branchdouble7x7_3 = nn.Sequential(
            nn.Conv2d(out_channels_7x7, out_channels_7x7, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d()
        )
        self.branchdouble7x7_4 = nn.Sequential(
            nn.Conv2d(out_channels_7x7, out_channels_7x7, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d()
        )
        self.branchdouble7x7_5 = nn.Sequential(
            nn.Conv2d(out_channels_7x7, 192, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d()
        )

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 192, kernel_size=1),
            nn.BatchNorm2d()
        )
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branchdouble7x7 = self.branchdouble7x7_1(x)
        branchdouble7x7 = self.branchdouble7x7_2(branchdouble7x7)
        branchdouble7x7 = self.branchdouble7x7_3(branchdouble7x7)
        branchdouble7x7 = self.branchdouble7x7_4(branchdouble7x7)
        branchdouble7x7 = self.branchdouble7x7_5(branchdouble7x7)
        branchpool = self.branchpool(x)
        outputs = [branch1x1, branch7x7, branchdouble7x7, branchpool]
        output = torch.cat(outputs, 1)
        return output


class Inceptionv3(nn.Module):
    def __init__(self, num_classes=2, init_weights=True):
        super(Inceptionv3, self).__init__()
        
        if init_weights:
            self._initialize_weights()

    def _make_layers(self, cfg):
        layers = []
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        output = 0
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

if __name__ == "__main__":
    inception = Inceptionv3()
    x = torch.zeros((1, 3, 224, 224))
    
    pred = inception(x)
    print(pred)