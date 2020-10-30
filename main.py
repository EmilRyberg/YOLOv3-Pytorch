import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import cv2 as cv
from PIL import Image
from model import Darknet53, YOLOv3, YOLOv3Tiny
from utils import non_max_suppression, rescale_boxes, load_classes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import random
import io


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


# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    #fig.savefig(buf, format="png", dpi=dpi)
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv.imdecode(img_arr, 1)
    #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    return img


def plot_detections(det, original_img):
    # Rescale boxes to original image
    #img = np.array(Image.open(path))
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    classes = load_classes("data/coco.names")
    print(det)
    if det is None:
        return None
    detections = rescale_boxes(det, 416, original_img.shape[:2])
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    bbox_colors = random.sample(colors, n_cls_preds)
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(original_img)

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
    buf = io.BytesIO()
    #fig.savefig(buf, format="png", dpi=dpi)
    plt.savefig(buf, bbox_inches="tight", pad_inches=0.0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv.imdecode(img_arr, 1)
    #img = None
    #plt.savefig(f"out.png", bbox_inches="tight", pad_inches=0.0)
    plt.close()
    return img


if __name__ == "__main__":
    dev = torch.device('cuda')
    cpu = torch.device('cpu')
    #darknet = Darknet53()
    #yolo = FullNet(80)
    yolo = YOLOv3Tiny(80)
    yolo.load_weights("yolov3-tiny.weights")
    yolo.eval()
    yolo = yolo.to(dev)
    img_path = "dog.jpg"
    cap = cv.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                    std=[0.229, 0.224, 0.225])])
        p_img = transforms.ToTensor()(img)
        p_img, _ = pad_to_square(p_img, 0)
        p_img = F.interpolate(p_img.unsqueeze(0), size=416, mode="nearest").squeeze(0)
        p_img = p_img.unsqueeze(0)

        p_img = p_img.to(dev)
        results = yolo(p_img)
        detections = non_max_suppression(results, 0.3)
        print(len(detections))

        img_s = plot_detections(detections[0], img)
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

