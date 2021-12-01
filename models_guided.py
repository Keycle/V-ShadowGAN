import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init
from cbam import *
from bam import *

class SENetBlock(nn.Module):
    def __init__(self, in_features, is_1x1conv=False):
        super(SENetBlock, self).__init__()
        self.is_1x1conv = is_1x1conv
        self.relu = nn.ReLU(inplace=True)
        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

        if is_1x1conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, in_features, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(in_features)
            )

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_features, in_features // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_features // 16, in_features, kernel_size=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x_shortcut = x
        x1 = self.conv_block(x)
        x2 = self.se(x1)
        x1 = x1 * x2
        if self.is_1x1conv:
            x_shortcut = self.shortcut(x_shortcut)
        x1 += x_shortcut
        x1 = self.relu(x1)
        return x1

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class CBAMBlock(nn.Module):
    expansion = 1

    def __init__(self, in_features):
        super(CBAMBlock, self).__init__()

        conv_block = [
            conv3x3(in_features, in_features, 1),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            conv3x3(in_features, in_features, 1),
            nn.InstanceNorm2d(in_features),

            CBAM(in_features, 16)
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class BAMBlock(nn.Module):
    def __init__(self, in_features):
        super(BAMBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)
        self.bam = BAM(in_features)


    def forward(self, x):
        x = x + self.conv_block(x)
        x = self.bam(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

""" Generator """
# Shadow Removal
class Generator_S2F(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator_S2F, self).__init__()



        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]


        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # SENet blocks
        for _ in range(n_residual_blocks):
             model += [SENetBlock(in_features)]

        # Upsampling + Conv
        out_features = in_features//2
        #for _ in range(2):
        #    model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
        #                nn.InstanceNorm2d(out_features),
        #                nn.ReLU(inplace=True) ]
        #    in_features = out_features
        #    out_features = in_features//2
        for _ in range(2):
            model += [  nn.Upsample(scale_factor=2),
			nn.Conv2d(in_features, out_features, 5, stride=1, padding=2),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2


        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7) ]
                    #nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return (self.model(x) + x).tanh()

# Shadow Generation
class Generator_F2S(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator_F2S, self).__init__()


        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # SENet blocks
        for _ in range(n_residual_blocks):
             model += [SENetBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        #for _ in range(2):
        #    model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
        #                nn.InstanceNorm2d(out_features),
        #                nn.ReLU(inplace=True) ]
        #    in_features = out_features
        #    out_features = in_features//2
        for _ in range(2):
            model += [  nn.Upsample(scale_factor=2),
			nn.Conv2d(in_features, out_features, 5, stride=1, padding=2),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7) ]
                    #nn.Tanh() ]

        self.model = nn.Sequential(*model)


    def forward(self, x):
        return (self.model(x) + x).tanh()

""" Discriminator """
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

