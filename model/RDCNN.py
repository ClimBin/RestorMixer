import torch.nn as nn
import torch.nn.functional as F
import torch


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, act=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if act:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
    
    
class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()

        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, act=True,bias=False,norm=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, act=True,bias=False,norm=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, act=True,bias=False,norm=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, act=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x

class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()

        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, act=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))
    

class ResidualDepthBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # (Depthwise + Pointwise)
        self.depthwise = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_channels,
            bias=False
        )
        self.pointwise = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.groups == m.out_channels:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.depthwise(out)
        out = self.pointwise(out)
        out = self.bn2(out)
        
        return F.gelu(out + shortcut)