import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import cv2 as cv
from PIL import Image
from model import YOLOv3, YOLOv3Tiny
from utils import non_max_suppression, rescale_boxes, load_classes
import matplotlib.pyplot as plt
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


def plot_detections(det, img):
    # Rescale boxes to original image
    img_cp = img.copy()
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    classes = load_classes("data/custom.names")
    #print(det)
    if det is None:
        return None
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
            cv.rectangle(img_cp, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv.putText(img_cp, classes[int(cls_pred)], (x1 + 20, y1 + 20), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return img_cp


if __name__ == "__main__":
    dev = torch.device('cuda')
    cpu = torch.device('cpu')
    #darknet = Darknet53()
    #yolo = YOLOv3(80)
    yolo = YOLOv3Tiny(11)
    yolo.load_weights("custom_yolov3-tiny.weights")
    yolo.eval()
    yolo = yolo.to(dev)
    img_path = "dog.jpg"
    cap = cv.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        p_img = transforms.ToTensor()(img)
        p_img, _ = pad_to_square(p_img, 0)
        p_img = F.interpolate(p_img.unsqueeze(0), size=416, mode="nearest").squeeze(0)
        p_img = p_img.unsqueeze(0)

        p_img = p_img.to(dev)
        results = yolo(p_img)
        detections = non_max_suppression(results, 0.3)

        img_s = plot_detections(detections[0], frame)
        #break
        if img_s is None:
            img_s = frame
        cv.imshow("det", img_s)

        # Display the resulting frame
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

