import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from model import Darknet53

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
    darknet = Darknet53()
    img = Image.open('dog.jpg')
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
    p_img = p_img#.to(dev)
    print('Img size 2: ', p_img.size())
    modl = list(list(darknet.children())[0].children())[0]
    #print(f"Weight: {modl.weight.data[0, 0, 0, 0]}")
    darknet.load_weights('darknet53_448.weights')
    #print(f"Weight after: {modl.weight.data[0, 0, 0, 0]}")
    #darknet#.to(dev)
    darknet.eval()
    results = darknet(p_img)
    data = results.data.numpy()
    argmax_pred = np.argmax(data, 1)
    max_pred = np.max(data, 1)
    print('Results size', results.size())
    print(f"Max index: {argmax_pred}, max_pred: {max_pred}")

