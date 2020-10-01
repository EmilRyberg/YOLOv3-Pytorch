import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from model import Darknet53

if __name__ == "__main__":
    darknet = Darknet53()
    img = Image.open('test_img.jpg')
    transform = transforms.Compose([transforms.ToTensor()])
    p_img = transform(img)
    print('Img size: ', p_img.size())
    p_img = torch.unsqueeze(p_img, 0)
    print('Img size 2: ', p_img.size())
    results = darknet(p_img)
    print('Results size', results.size())

