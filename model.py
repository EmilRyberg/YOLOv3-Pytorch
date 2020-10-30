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

    def load_weights(self, current_ptr, weights):
        """Parses and loads the weights stored in 'weights_path'"""

        # Establish cutoff for loading backbone weights
        cutoff = None

        ptr = current_ptr
        for i, module in enumerate(self.module_list):
           ptr = module.load_weights(ptr, weights)
        return ptr


class ResidualBlock(nn.Module):
    def __init__(self, channels_in):
        super(ResidualBlock, self).__init__()
        first_out_channels = channels_in // 2
        self.module_list = nn.ModuleList([
            ConvBlock(channels_in, first_out_channels, kernel_size=1),
            ConvBlock(first_out_channels, first_out_channels * 2, kernel_size=3, padding=1)
        ])

    def forward(self, x):
        skip_connection = x
        layer_outputs = []
        for i, module in enumerate(self.module_list):
            x = module(x)
            layer_outputs.append(x)
        x = x + skip_connection
        layer_outputs.append(x)
        return x, layer_outputs

    def load_weights(self, current_ptr, weights):
        ptr = current_ptr
        for i, module in enumerate(self.module_list):
            ptr = module.load_weights(ptr, weights)
        return ptr


class YOLOTinyBackbone(nn.Module):
    def __init__(self):
        super(YOLOTinyBackbone, self).__init__()
        self.module_list = nn.ModuleList([
            ConvBlock(3, 16, 3, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(16, 32, 3, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64, 3, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128, 3, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256, 3, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(256, 512, 3, padding=1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(2, 1),
            ConvBlock(512, 1024, 3, padding=1)]
        )

    def forward(self, x):
        layer_outputs = []
        for i, module in enumerate(self.module_list):
            x = module(x)
            layer_outputs.append(x)
        return x, layer_outputs

    def load_weights(self, current_ptr, weights):
        ptr = current_ptr
        for i, module in enumerate(self.module_list):
            if type(module) == nn.MaxPool2d or type(module) == nn.ZeroPad2d:
                continue
            ptr = module.load_weights(ptr, weights)
        return ptr


class YOLOv3Tiny(nn.Module):
    def __init__(self, num_classes=80, img_size=416):
        super(YOLOv3Tiny, self).__init__()
        self.img_size = img_size
        self.backbone = YOLOTinyBackbone()
        self.cnn_list = nn.ModuleList([
            ConvBlock(1024, 256, 1),
            ConvBlock(256, 512, 3, padding=1),
            ConvBlock(512, (num_classes+5)*3, 1, batch_norm=False, activation=False)
        ])
        self.first_yolo = YOLOLayer([(81, 82),
                                     (135, 169),
                                     (344, 319)], num_classes, img_size)
        self.second_yolo_conv = ConvBlock(256, 128, kernel_size=1)
        self.upsample = Upsample(2)
        self.second_yolo_conv2 = ConvBlock(384, 256, 3, padding=1)
        self.second_yolo_conv3 = ConvBlock(256, (num_classes+5)*3, 1, batch_norm=False, activation=False)
        self.second_yolo = YOLOLayer([(10, 14),
                                     (23, 27),
                                     (37, 58)], num_classes, img_size)

    def forward(self, x):
        layer_outputs = []
        yolo_outputs = []
        x, backbone_outputs = self.backbone(x)
        layer_outputs.extend(backbone_outputs)
        for module in self.cnn_list:
            x = module(x)
            layer_outputs.append(x)
        x, layer_loss = self.first_yolo(x)
        layer_outputs.append(x)
        yolo_outputs.append(x)
        route_output = layer_outputs[-4]
        x = torch.cat([route_output], 1)
        layer_outputs.append(x)
        x = self.second_yolo_conv(x)
        layer_outputs.append(x)
        x = self.upsample(x)
        layer_outputs.append(x)
        x = torch.cat([x, layer_outputs[8]], 1)
        layer_outputs.append(x)
        x = self.second_yolo_conv2(x)
        layer_outputs.append(x)
        x = self.second_yolo_conv3(x)
        layer_outputs.append(x)
        x, layer2_loss = self.second_yolo(x)
        layer_outputs.append(x)
        yolo_outputs.append(x)
        return to_cpu(torch.cat(yolo_outputs, 1))

    def load_weights(self, weights_path):
        ptr = 0
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        ptr = self.backbone.load_weights(ptr, weights)
        for i, module in enumerate(self.cnn_list):
            ptr = module.load_weights(ptr, weights)
        ptr = self.second_yolo_conv.load_weights(ptr, weights)
        ptr = self.second_yolo_conv2.load_weights(ptr, weights)
        ptr = self.second_yolo_conv3.load_weights(ptr, weights)
        print(f"Loaded weights {ptr}/{weights.shape[0]}")


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

        #self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)
        #print(f"YOLO initial: {x.shape}")
        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
        )
        #print(f"After: {prediction.shape}")

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
        self.conv = nn.Conv2d(in_channels=channels_in,
                      out_channels=channels_out,
                      kernel_size=kernel_size, bias=not batch_norm, padding=padding, stride=stride)
        self.has_batch_norm = False
        self.batch_norm = None

        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(channels_out, momentum=0.9, eps=1e-5)
            self.has_batch_norm = True

        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.has_batch_norm:
            x = self.batch_norm(x)
        if self.activation:
            x = nn.LeakyReLU(0.1)(x)
        return x

    def load_weights(self, current_ptr, weights):
        ptr = current_ptr
        conv_layer = self.conv
        if self.has_batch_norm:
            # Load BN bias, weights, running mean and running variance
            bn_layer = self.batch_norm
            num_b = bn_layer.bias.numel()  # Number of biases
            # Bias
            bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
            bn_layer.bias.data.copy_(bn_b)
            ptr += num_b
            # Weight
            bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
            bn_layer.weight.data.copy_(bn_w)
            ptr += num_b
            # Running Mean
            bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
            bn_layer.running_mean.data.copy_(bn_rm)
            ptr += num_b
            # Running Var
            bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
            bn_layer.running_var.data.copy_(bn_rv)
            ptr += num_b
        else:
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
        return ptr


class YOLOv3(nn.Module):
    def __init__(self, num_classes, img_dim=416):
        super(YOLOv3, self).__init__()
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
        self.second_yolo_conv_blocks = nn.ModuleList([
            ConvBlock(768, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3, padding=1),
            ConvBlock(512, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3, padding=1),
            ConvBlock(512, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3, padding=1),
            ConvBlock(512, 255, kernel_size=1, batch_norm=False, activation=False),
        ])
        self.second_yolo = YOLOLayer([(30, 61),
                                     (62, 45),
                                     (59, 119)], num_classes, img_dim)
        self.conv2 = ConvBlock(256, 128, kernel_size=1)
        self.scale2 = Upsample(2)
        self.third_yolo_conv_blocks = nn.ModuleList([
            ConvBlock(384, 128, kernel_size=1),
            ConvBlock(128, 256, kernel_size=3, padding=1),
            ConvBlock(256, 128, kernel_size=1),
            ConvBlock(128, 256, kernel_size=3, padding=1),
            ConvBlock(256, 128, kernel_size=1),
            ConvBlock(128, 256, kernel_size=3, padding=1),
            ConvBlock(256, 255, kernel_size=1, batch_norm=False, activation=False),
        ])
        self.third_yolo = YOLOLayer([(10, 13),
                                     (16, 30),
                                     (33, 23)], num_classes, img_dim)

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
        print(f"Yolo dim {x.size()}")
        route_output = layer_outputs[-4]
        x = torch.cat([route_output], 1)
        print(f"First route dim {x.size()}")
        layer_outputs.append(x)
        x = self.conv1(x)
        layer_outputs.append(x)
        x = self.scale1(x)
        layer_outputs.append(x)
        x = torch.cat([x, layer_outputs[61]], 1)
        print(f"Second route dim {x.size()}")
        layer_outputs.append(x)
        for i, module in enumerate(self.second_yolo_conv_blocks):
            x = module(x)
            layer_outputs.append(x)
        x, layer2_loss = self.second_yolo(x)
        layer_outputs.append(x)
        yolo_outputs.append(x)
        print(f"Yolo 2 dim {x.size()}")
        route_output = layer_outputs[-4]
        x = torch.cat([route_output], 1)
        print(f"Third route dim {x.size()}")
        layer_outputs.append(x)
        x = self.conv2(x)
        layer_outputs.append(x)
        x = self.scale2(x)
        layer_outputs.append(x)
        x = torch.cat([x, layer_outputs[36]], 1)
        print(f"Fourth route dim {x.size()}")
        layer_outputs.append(x)
        for i, module in enumerate(self.third_yolo_conv_blocks):
            x = module(x)
            layer_outputs.append(x)
        x, layer3_loss = self.third_yolo(x)
        yolo_outputs.append(x)
        #print(f"Yolo output: {x.shape}")
        layer_outputs.append(x)
        #print(f"YOLO before: {yolo_outputs}")
        return to_cpu(torch.cat(yolo_outputs, 1))

    def load_weights(self, weights_path):
        ptr = 0
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        ptr = self.darknet.load_weights(ptr, weights)
        for i, module in enumerate(self.first_path):
            ptr = module.load_weights(ptr, weights)
        ptr = self.conv1.load_weights(ptr, weights)
        for i, module in enumerate(self.second_yolo_conv_blocks):
            ptr = module.load_weights(ptr, weights)
        ptr = self.conv2.load_weights(ptr, weights)
        for i, module in enumerate(self.third_yolo_conv_blocks):
            ptr = module.load_weights(ptr, weights)
        print(f"Loaded weights {ptr}/{weights.shape[0]}")
