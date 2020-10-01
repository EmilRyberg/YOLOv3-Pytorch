import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from model import Darknet53

if __name__ == "__main__":
    darknet = Darknet53()
    img = Image.open('dog_small.jpg')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
    p_img = transforms.ToTensor()(img)
    print(p_img.max())
    print('Img size: ', p_img.size())
    p_img = torch.unsqueeze(p_img, 0)
    print('Img size 2: ', p_img.size())
    modl = list(list(darknet.children())[0].children())[0]
    print(f"Weight: {modl.weight.data[0, 0, 0, 0]}")
    darknet.load_weights('darknet53.weights')
    print(f"Weight after: {modl.weight.data[0, 0, 0, 0]}")
    results = darknet(p_img)
    data = results.data.numpy()
    argmax_pred = np.argmax(data, 1)
    max_pred = np.max(data, 1)
    print('Results size', results.size())
    print(f"Max index: {argmax_pred}, max_pred: {max_pred}")

