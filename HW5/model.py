# Adityan Jothi
# USC ID 8162222801
# jothi@usc.edu

import torch
import torch.nn
from torch.nn import Conv2d as ConvLayer
from torch.nn import MaxPool2d as MaxPoolLayer
from torch.nn import Linear as FullyConnectedLayer
from torch.nn import BatchNorm2d as BN2dLayer
from torch.nn import Module as Parent
from torch import flatten as FlattenLayer
from torch.nn import ReLU as ActReLU
from torch.nn import LeakyReLU as ActLReLU
from torch.nn.init import uniform_ as UniformInitializer
from torch.nn.init import normal_ as NormalInitializer
from torch.nn.init import xavier_uniform_ as XavierUniformInitializer
from torch.nn.init import xavier_normal_ as XavierNormalInitializer
from torch.nn.init import kaiming_uniform_ as KaimingUniformInitializer
from torch.nn.init import kaiming_normal_ as KaimingNormalInitializer

#LeNet Implementation for problem 1b
class ClassifierLN5(Parent):
    def __init__(self, in_channels=1, number_of_classes= 10, init_method=None):
        super(ClassifierLN5, self).__init__()
        self.number_of_classes = number_of_classes
        self.activation = ActReLU()

        self.layer_1 = ConvLayer(in_channels, 6, 5)
        self.layer_2 = MaxPoolLayer(2)
        self.layer_3 = ConvLayer(6, 16, 5)
        self.layer_4 = MaxPoolLayer(2)
        self.layer_5 = FullyConnectedLayer(400, 120)
        self.layer_6 = FullyConnectedLayer(120, 84)
        self.layer_7 = FullyConnectedLayer(84, number_of_classes)


        to_be_filled = [self.layer_1.weight,
            self.layer_3.weight,
            self.layer_5.weight,
            self.layer_6.weight ,
            self.layer_7.weight]
        
        for param in to_be_filled:
            if init_method=='uniform':
                UniformInitializer(param)
            elif init_method=='normal':
                NormalInitializer(param)
            elif init_method=='xavier_uniform':
                XavierUniformInitializer(param)
            elif init_method=='xavier_normal':
                XavierNormalInitializer(param)
            elif init_method=='he_uniform':
                KaimingUniformInitializer(param)
            elif init_method=='he_normal':
                KaimingNormalInitializer(param)

    def forward(self, x):
        if x.size(-1)==28:
            x = torch.nn.ZeroPad2d(2)(x)
        x = self.layer_1(x)
        x = self.activation(self.layer_2(x))
        x = self.layer_3(x)
        x = self.activation(self.layer_4(x))
        x = x.view(x.size(0),-1)
        x = self.activation(self.layer_5(x))
        x = self.activation(self.layer_6(x))
        x = self.layer_7(x)
        
        return x
        
#Custom net for problem1c
class CustomClassifier(Parent):
    def __init__(self, in_channels=1, number_of_classes=10, init_method=None):
        super(CustomClassifier, self).__init__()
        self.number_of_classes = number_of_classes
        self.in_channels = in_channels

        self.activation = ActLReLU()

        self.layer_1 = ConvLayer(in_channels, 8, 3)
        self.layer_2 = ConvLayer(8, 16, 3, stride=2)
        self.layer_3 = ConvLayer(16, 32, 3)
        self.layer_4 = MaxPoolLayer(2)
        self.bn_layer_1 = BN2dLayer(32)
        self.layer_5 = ConvLayer(32, 64, 3)
        self.layer_6 = MaxPoolLayer(2)
        self.bn_layer_2 = BN2dLayer(64)
        self.layer_7 = FullyConnectedLayer(64, number_of_classes)

    
    def forward(self, x):
        x = self.activation(self.layer_1(x))
        x = self.activation(self.layer_2(x))
        x = self.bn_layer_1(self.layer_3(x))
        x = self.layer_4(x)
        x = self.activation(self.bn_layer_2(self.layer_5(x)))
        x = self.layer_6(x)
        x = x.view(x.size(0),-1)

        return self.layer_7(x)
