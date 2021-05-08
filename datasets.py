#%%
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from PIL import ImageDraw




# json_file = '../data/mydata/all.json'



class GFKDtrack_dataset(Dataset):
    def __init__(self, data_dir, json_file):
        # imgs = os.listdir(os.path.join(data_dir, 'images'))
        self.labels, img_names = self.get_labels(os.path.join(data_dir, json_file))
        self.imgs = [os.path.join(data_dir, 'images', x) for x in img_names]
        # self.labels =

    def get_labels(self, json_file):
        with open(json_file, 'r') as f:
            all_label = json.load(f)
        labels = []
        img_names = list(all_label.keys())

        for imgn in img_names:
            name = imgn.split('/')[-1]
            # assert name in all_label.keys()
            xy = all_label[name]['xy']
            x = int(xy[0])
            y = int(xy[1])
            labels.append([x, y])

            # print(key, ': ', x, y)
        return labels, img_names
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        x, y = self.labels[index]
        # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = Image.open(img_path)

        # x = x / img.width
        # y = y / img.height

        label = torch.Tensor([x, y]).float()
        mytrans = transforms.ToTensor()
        img = mytrans(img)
        return img, label
#%%

if __name__ == "__main__":
    #%%
    data_dir = '../data/mydata'
    json_file = 'test.json'
    mydata = GFKDtrack_dataset(data_dir, json_file)
    #%%
    to_img = transforms.ToPILImage()
    d = mydata[4]
    img = to_img(d[0])

    draw = ImageDraw.ImageDraw(img)
    x = int(d[1][0].item()*1024)
    y = int(d[1][1].item()*1024)
    draw.rectangle((x-10, y-10, x+10, y+10))

    img.show()

    # for i in range(10):
    #     print(mydata[i])
