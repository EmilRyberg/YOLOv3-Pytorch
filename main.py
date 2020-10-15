import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from model import Darknet53, FullNet
from utils import non_max_suppression, rescale_boxes, load_classes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import random


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


if __name__ == "__main__":
    dev = torch.device('cuda')
    cpu = torch.device('cpu')
    #darknet = Darknet53()
    yolo = FullNet(80)
    img_path = "dog.jpg"
    img = Image.open(img_path)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
    transform2 = transforms.Compose([
        transforms.CenterCrop(448),
        transforms.ToTensor()
    ])
    p_img = transforms.ToTensor()(img)
    p_img, _ = pad_to_square(p_img, 0)
    p_img = F.interpolate(p_img.unsqueeze(0), size=448, mode="nearest").squeeze(0)
    p_img = p_img.unsqueeze(0)
    print(p_img.max())
    print(p_img.min())
    print('Img size: ', p_img.size())
    #p_img = torch.unsqueeze(p_img, 0)
    p_img = p_img.to(dev)
    print('Img size 2: ', p_img.size())
    yolo.load_weights("yolov3.weights")
    yolo.eval()
    yolo = yolo.to(dev)
    results = yolo(p_img)
    detections = non_max_suppression(results)
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    classes = load_classes("data/coco.names")

    img = np.array(Image.open(img_path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    img_detections = []
    img_detections.extend(detections)
    print(f"Shape1: {img_detections}, 2: {detections}")

    for det in img_detections:
        # Rescale boxes to original image
        detections = rescale_boxes(det, 416, img.shape[:2])
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)

        if det is not None:
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig(f"out.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()

