import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.module_list = nn.ModuleList([
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            ResidualBlock(64, 32),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            ResidualBlock(128, 64),
            ResidualBlock(128, 64),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1)
        ])

        for i in range(8):
            self.module_list.append(ResidualBlock(256, 128))
        self.module_list.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2, bias=False))
        self.module_list.append(nn.BatchNorm2d(512))
        self.module_list.append(nn.LeakyReLU(0.1))
        for i in range(8):
            self.module_list.append(ResidualBlock(512, 256))
        self.module_list.append(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=2, bias=False))
        self.module_list.append(nn.BatchNorm2d(1024))
        self.module_list.append(nn.LeakyReLU(0.1))
        for i in range(4):
            self.module_list.append(ResidualBlock(1024, 512))
        self.module_list.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.linear = nn.Linear(1024, 1000)

    def forward(self, x):
        for i, module in enumerate(self.module_list):
            print(f"layer {i} size: {x.size()}")
            x = module(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.linear(x)
        x = nn.Softmax(1)(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def load_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            # self.header_info = header  # Needed to write header when saving weights
            # self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights
            print('weights', weights)

        # Establish cutoff for loading backbone weights
        cutoff = None
        # if "darknet53.conv.74" in weights_path:
        #    cutoff = 75

        ptr = 0
        layers = 0
        for i, module in enumerate(self.module_list):
            if i == cutoff:
                break
            if type(module) == ResidualBlock:
                for inner_module in module.sequential.modules():
                    print(f"{i} loading for ResidualBlock")
                    ptr = load_weights_for_module(inner_module, ptr, weights, log=False)
                    if type(inner_module) == nn.Conv2d:
                        layers += 1
            else:
                print(f"{i} normal block")
                ptr = load_weights_for_module(module, ptr, weights)
                if type(module) == nn.Conv2d:
                    layers += 1
        ptr = load_weights_for_module(self.linear, ptr, weights)
        layers += 1
        print(f"Loaded {layers} layers")
        print(f"PTR: {ptr}, weights: {weights.shape}")


def load_weights_for_module(module, ptr, weights, log=False):
    l_ptr = ptr
    if log:
        print(f"Module: {module}")
    if type(module) == nn.Conv2d:
        print('Is conv')
        #print(f"Module: {module}")
        conv_layer = module
        num_w = conv_layer.weight.numel()
        conv_w = torch.from_numpy(weights[l_ptr: l_ptr + num_w]).view_as(conv_layer.weight)
        conv_layer.weight.data.copy_(conv_w)
        l_ptr += num_w
    elif type(module) == nn.BatchNorm2d:
        print('Is batch norm')
        # Load BN bias, weights, running mean and running variance
        bn_layer = module
        num_b = bn_layer.bias.numel()  # Number of biases
        # Bias
        bn_b = torch.from_numpy(weights[l_ptr: l_ptr + num_b]).view_as(bn_layer.bias)
        bn_layer.bias.data.copy_(bn_b)
        l_ptr += num_b
        # Weight
        bn_w = torch.from_numpy(weights[l_ptr: l_ptr + num_b]).view_as(bn_layer.weight)
        bn_layer.weight.data.copy_(bn_w)
        l_ptr += num_b
        # Running Mean
        bn_rm = torch.from_numpy(weights[l_ptr: l_ptr + num_b]).view_as(bn_layer.running_mean)
        bn_layer.running_mean.data.copy_(bn_rm)
        l_ptr += num_b
        # Running Var
        bn_rv = torch.from_numpy(weights[l_ptr: l_ptr + num_b]).view_as(bn_layer.running_var)
        bn_layer.running_var.data.copy_(bn_rv)
        l_ptr += num_b
        # else:
        # Load conv. bias
        # num_b = conv_layer.bias.numel()
        # conv_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
        # conv_layer.bias.data.copy_(conv_b)
        # ptr += num_b
        # Load conv. weights
    elif type(module) == nn.Linear:
        print('is linear')
        num_b = module.bias.numel()
        l_b = torch.from_numpy(weights[l_ptr: l_ptr + num_b]).view_as(module.bias)
        module.bias.data.copy_(l_b)
        l_ptr += num_b
        num_w = module.weight.numel()
        l_w = torch.from_numpy(weights[l_ptr: l_ptr + num_w]).view_as(module.weight)
        module.weight.data.copy_(l_w)
        l_ptr += num_w
    return l_ptr


class ResidualBlock(nn.Module):
    def __init__(self, channels_in, first_out_channels):
        super(ResidualBlock, self).__init__()
        self.channels_in = channels_in
        self.first_out_channels = first_out_channels
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=channels_in,
                      out_channels=first_out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(first_out_channels, momentum=0.9, eps=1e-5),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=self.first_out_channels,
                      out_channels=self.first_out_channels * 2,
                      kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(first_out_channels * 2, momentum=0.9, eps=1e-5),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        skip_connection = x
        x = self.sequential(x)
        x = x + skip_connection
        return x
