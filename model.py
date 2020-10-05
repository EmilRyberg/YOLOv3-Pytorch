import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.module_list = nn.ModuleList([
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.9),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(0.1),
            ResidualBlock(64, 32),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1),
            ResidualBlock(128, 64),
            ResidualBlock(128, 64),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.LeakyReLU(0.1)
        ])

        for i in range(8):
            self.module_list.append(ResidualBlock(256, 128))
        self.module_list.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2, bias=False))
        self.module_list.append(nn.BatchNorm2d(512, momentum=0.9))
        self.module_list.append(nn.LeakyReLU(0.1))
        for i in range(8):
            self.module_list.append(ResidualBlock(512, 256))
        self.module_list.append(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=2, bias=False))
        self.module_list.append(nn.BatchNorm2d(1024, momentum=0.9))
        self.module_list.append(nn.LeakyReLU(0.1))
        for i in range(4):
            self.module_list.append(ResidualBlock(1024, 512))
        self.module_list.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.linear = nn.Linear(1024, 1000)
        #self.module_list.append(nn.Conv2d(in_channels=1024, out_channels=1000, kernel_size=1))

    def forward(self, x):
        for i, module in enumerate(self.module_list):
            #print(f"layer {i} size: {x.size()}")
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
            print('weights', weights.shape)

        # Establish cutoff for loading backbone weights
        cutoff = None
        # if "darknet53.conv.74" in weights_path:
        #    cutoff = 75

        ptr = 0
        layers = 0
        current_conv = None
        current_bn = None
        for i, module in enumerate(self.module_list):
            #print('m', type(module))
            if i == cutoff:
                break
            if type(module) == ResidualBlock:
                rb_conv = None
                rb_bn = None
                for inner_module in module.sequential.modules():
                    if type(inner_module) != nn.Conv2d and type(inner_module) != nn.BatchNorm2d:
                        continue
                    if type(inner_module) == nn.Conv2d:
                        rb_conv = inner_module
                        layers += 1
                        continue
                    if type(inner_module) == nn.BatchNorm2d:
                        rb_bn = inner_module
                    #print('type', type(inner_module))
                    #print(f"{i} loading for ResidualBlock")
                    #w = inner_module.weight.data[0, 0, 0, 0].clone()
                    ptr = load_weights_for_module(rb_conv, rb_bn, ptr, weights)
                    #w2 = inner_module.weight.data[0, 0, 0, 0]
                    #print(f"was: {w}, is: {w2}")
            else:
                #print(f"{i} normal block")
                # is_final = False
                # if i == len(self.module_list) - 1:
                #     is_final = True
                if type(module) != nn.Conv2d and type(module) != nn.BatchNorm2d:
                    continue
                if type(module) == nn.Conv2d:
                    current_conv = module
                    layers += 1
                    continue
                    # if not is_final:
                    #     continue
                if type(module) == nn.BatchNorm2d:
                    current_bn = module
                ptr = load_weights_for_module(current_conv, current_bn, ptr, weights)
                current_conv = None
                current_bn = None

        #ptr = load_weights_for_module(self.linear, ptr, weights)
        layers += 1
        linear_module = self.linear
        num_b = linear_module.bias.numel()
        l_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(linear_module.bias)
        linear_module.bias.detach().copy_(l_b)
        ptr += num_b
        num_w = linear_module.weight.numel()
        l_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(linear_module.weight)
        linear_module.weight.detach().copy_(l_w)
        ptr += num_w
        print(f"Loaded {layers} layers")
        print(f"PTR: {ptr}, weights: {weights.shape}")


def load_weights_for_module(current_conv, current_bn, ptr, weights):
    l_ptr = ptr
    # Load BN bias, weights, running mean and running variance
    bn_layer = current_bn
    num_b = bn_layer.bias.numel()  # Number of biases
    # Bias
    bn_b = torch.from_numpy(weights[l_ptr: l_ptr + num_b]).view_as(bn_layer.bias)
    bn_layer.bias.detach().copy_(bn_b)
    l_ptr += num_b
    # Weight
    bn_w = torch.from_numpy(weights[l_ptr: l_ptr + num_b]).view_as(bn_layer.weight)
    bn_layer.weight.detach().copy_(bn_w)
    l_ptr += num_b
    # Running Mean
    bn_rm = torch.from_numpy(weights[l_ptr: l_ptr + num_b]).view_as(bn_layer.running_mean)
    bn_layer.running_mean.detach().copy_(bn_rm)
    l_ptr += num_b
    # Running Var
    bn_rv = torch.from_numpy(weights[l_ptr: l_ptr + num_b]).view_as(bn_layer.running_var)
    bn_layer.running_var.detach().copy_(bn_rv)
    l_ptr += num_b

    #Load conv
    conv_layer = current_conv
    # if is_final:
    #     # Load conv. bias
    #     num_b = conv_layer.bias.numel()
    #     conv_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
    #     conv_layer.bias.detach().copy_(conv_b)
    #     ptr += num_b
    # print('Is conv')
    # print(f"Module: {module}")
    num_w = conv_layer.weight.numel()
    conv_w = torch.from_numpy(weights[l_ptr: l_ptr + num_w]).view_as(conv_layer.weight)
    #print('weights', conv_w)
    conv_layer.weight.detach().copy_(conv_w)
    l_ptr += num_w

    # elif type(module) == nn.Linear:
    #     print('is linear')
    #     num_b = module.bias.numel()
    #     l_b = torch.from_numpy(weights[l_ptr: l_ptr + num_b]).view_as(module.bias)
    #     module.bias.detach().copy_(l_b)
    #     l_ptr += num_b
    #     num_w = module.weight.numel()
    #     l_w = torch.from_numpy(weights[l_ptr: l_ptr + num_w]).view_as(module.weight)
    #     module.weight.detach().copy_(l_w)
    #     l_ptr += num_w
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
