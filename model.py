import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.modules = [

        ]

    def forward(self, x):
        x = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)(x)
        x = nn.LeakyReLU(0.1)(x)
        x = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2)(x)
        x = nn.LeakyReLU(0.1)(x)
        x = ResidualBlock(x, 64, 32)(x)
        x = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2)(x)
        x = nn.LeakyReLU(0.1)(x)
        x = ResidualBlock(x, 128, 64)(x)
        x = ResidualBlock(x, 128, 64)(x)
        x = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2)(x)
        x = nn.LeakyReLU(0.1)(x)
        for i in range(8):
            x = ResidualBlock(x, 256, 128)(x)
        x = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2)(x)
        nn.LeakyReLU(0.1)(x)
        for i in range(8):
            x = ResidualBlock(x, 512, 256)(x)
        x = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=2)(x)
        x = nn.LeakyReLU(0.1)(x)
        for i in range(4):
            x = ResidualBlock(x, 1024, 512)(x)
        x = nn.AdaptiveAvgPool2d(1)(x)
        print('avg pool size', x.size())
        x = x.view(-1, self.num_flat_features(x))
        x = nn.Linear(1024, 1000)(x)
        x = nn.Softmax(-1)(x)
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
            #self.header_info = header  # Needed to write header when saving weights
            #self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        #if "darknet53.conv.74" in weights_path:
        #    cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # if module_def["batch_normalize"]:
                #     # Load BN bias, weights, running mean and running variance
                #     bn_layer = module[1]
                #     num_b = bn_layer.bias.numel()  # Number of biases
                #     # Bias
                #     bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                #     bn_layer.bias.data.copy_(bn_b)
                #     ptr += num_b
                #     # Weight
                #     bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                #     bn_layer.weight.data.copy_(bn_w)
                #     ptr += num_b
                #     # Running Mean
                #     bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                #     bn_layer.running_mean.data.copy_(bn_rm)
                #     ptr += num_b
                #     # Running Var
                #     bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                #     bn_layer.running_var.data.copy_(bn_rv)
                #     ptr += num_b
                # else:
                    # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

class ResidualBlock(nn.Module):
    def __init__(self, skip_output, channels_in, first_out_channels):
        super(ResidualBlock, self).__init__()
        self.skip_output = skip_output
        self.channels_in = channels_in
        self.first_out_channels = first_out_channels

    def forward(self, x):
        x = nn.Conv2d(in_channels=self.channels_in,
                      out_channels=self.first_out_channels,
                      kernel_size=1)(x)
        x = nn.LeakyReLU(0.1)(x)
        x = nn.Conv2d(in_channels=self.first_out_channels,
                      out_channels=self.first_out_channels * 2,
                      kernel_size=3,
                      padding=1)(x)
        x = nn.LeakyReLU(0.1)(x)
        x = x + self.skip_output
        return x