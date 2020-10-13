import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import non_max_suppression, build_targets, to_cpu


class Darknet53(nn.Module):
    def __init__(self, with_head=False):
        super(Darknet53, self).__init__()
        if with_head:
            print("Warning: Darknet53 used with with_head=True. This might cause problems :-)")
        self.with_head = with_head
        self.module_list = nn.ModuleList([
            ConvBlock(3, 32, kernel_size=3, padding=1),
            ConvBlock(32, 64, kernel_size=3, padding=1, stride=2),
            ResidualBlock(64),
            ConvBlock(64, 128, kernel_size=3, padding=1, stride=2),
            ResidualBlock(128),
            ResidualBlock(128),
            ConvBlock(128, 256, kernel_size=3, padding=1, stride=2)
        ])

        for i in range(8):
            self.module_list.append(ResidualBlock(256))
        self.module_list.append(ConvBlock(256, 512, kernel_size=3, padding=1, stride=2))
        for i in range(8):
            self.module_list.append(ResidualBlock(512))
        self.module_list.append(ConvBlock(512, 1024, kernel_size=3, padding=1, stride=2))
        for i in range(4):
            self.module_list.append(ResidualBlock(1024))
        if with_head:
            self.module_list.append(nn.AdaptiveAvgPool2d((1, 1)))
            self.module_list.append(nn.Linear(1024, 1000))

    def forward(self, x):
        layer_outputs = []
        for i, module in enumerate(self.module_list):
            if type(module) == ResidualBlock:
                x, residual_layer_outputs = module(x)
                layer_outputs.extend(residual_layer_outputs)
            else:
                x = module(x)
                layer_outputs.append(x)

        if self.with_head:
            x = x.view(-1, self.num_flat_features(x))
            x = self.linear(x)
            x = nn.Softmax(1)(x)
        return x, layer_outputs

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

        ptr = 0
        layers = 0
        current_conv = None
        current_bn = None
        for i, module in enumerate(self.module_list):
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
                    ptr = load_weights_for_module(rb_conv, rb_bn, ptr, weights)
            else:
                if type(module) != nn.Conv2d and type(module) != nn.BatchNorm2d:
                    continue
                if type(module) == nn.Conv2d:
                    current_conv = module
                    layers += 1
                    continue
                if type(module) == nn.BatchNorm2d:
                    current_bn = module
                ptr = load_weights_for_module(current_conv, current_bn, ptr, weights)
                current_conv = None
                current_bn = None

        # ptr = load_weights_for_module(self.linear, ptr, weights)
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

    # Load conv
    conv_layer = current_conv
    num_w = conv_layer.weight.numel()
    conv_w = torch.from_numpy(weights[l_ptr: l_ptr + num_w]).view_as(conv_layer.weight)
    conv_layer.weight.detach().copy_(conv_w)
    l_ptr += num_w
    return l_ptr


class ResidualBlock(nn.Module):
    def __init__(self, channels_in):
        super(ResidualBlock, self).__init__()
        first_out_channels = channels_in // 2
        self.modules = nn.ModuleList([
            ConvBlock(channels_in, first_out_channels, kernel_size=1),
            ConvBlock(first_out_channels, first_out_channels * 2, kernel_size=1, padding=1)
        ])

    def forward(self, x):
        skip_connection = x
        layer_outputs = []
        for i, module in enumerate(self.modules):
            x = module(x)
            layer_outputs.append(x)
        x = x + skip_connection
        layer_outputs.append(x)
        return x, layer_outputs


class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size
            }

            return output, total_loss


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class ConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, padding=0, batch_norm=True, activation=True, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.ModuleList([
            nn.Conv2d(in_channels=channels_in,
                      out_channels=channels_out,
                      kernel_size=kernel_size, bias=not batch_norm, padding=padding, stride=stride)]
        )
        if batch_norm:
            self.conv.append(nn.BatchNorm2d(channels_out, momentum=0.9, eps=1e-5))

        if activation:
            self.conv.append(nn.LeakyReLU(0.1))

    def forward(self, x):
        for i, module in enumerate(self.conv):
            x = module(x)
        return x


class FullNet(nn.Module):
    def __init__(self, num_classes, img_dim=416):
        super(FullNet, self).__init__()
        self.darknet = Darknet53()
        self.first_path = nn.ModuleList([
            ConvBlock(1024, 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 255, kernel_size=1, batch_norm=False, activation=False),
        ])
        self.first_yolo = YOLOLayer([(116, 90),
                                     (156, 198),
                                     (373, 326)], num_classes, img_dim)
        self.conv1 = ConvBlock(512, 256, kernel_size=1)
        self.scale1 = Upsample(2)
        # self.second_yolo_conv_blocks = nn.ModuleList([
        #     
        # ])

    def forward(self, x):
        yolo_outputs = []
        layer_outputs = []
        x, darknet_outputs = self.darknet(x)
        layer_outputs.extend(darknet_outputs)
        for i, module in enumerate(self.first_path):
            x = module(x)
            layer_outputs.append(x)
        x, layer_loss = self.first_yolo(x)
        layer_outputs.append(x)
        yolo_outputs.append(x)
        route_output = layer_outputs[-4]
        x = torch.cat([route_output], 1)
        layer_outputs.append(x)
        x = self.conv1(x)
        layer_outputs.append(x)
        x = self.scale1(x)
        layer_outputs.append(x)
        x = torch.cat([x, layer_outputs[61]])
        layer_outputs.append(x)
        x = Conv

