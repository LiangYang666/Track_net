import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

imgs_dir = '../../data/mydata/images'
json_file = '../../data/mydata/all.json'
img_w = 1024
img_h = 1024
N = 50

if __name__ == "__main__":
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir)
    labels = {}
    radious = 10
    for i in tqdm(range(N)):
        (x, y) = np.random.randint(0, min(img_w, img_h), size=2, dtype=int)
        img = np.random.randint(0, 200, (img_w, img_h), dtype=np.uint8)
        cv2.circle(img, (x, y), radious, color=255, thickness=-1)
        # plt.imshow(img, cmap='gray')
        # plt.show()
        name = f'{i}.png'
        cv2.imwrite(os.path.join(imgs_dir, name), img)
        labels[name] = {'xy': [int(x), int(y)], 'radious': int(radious)}

    with open(json_file, 'w') as f:
        json.dump(labels, f, indent=2)
    # plt.plot()