from net import MyNet, MyNet1
from datasets import GFKDtrack_dataset
from torch.utils.data import DataLoader
# import torch.nn as nn
import torch
from tqdm import tqdm
import logging
import os
import sys
import ipdb
import numpy as np

data_dir = '../data/mydata'

def test(data_dir, ckp_file, test_file, device):
    model = MyNet()
    model.to(device=device)
    ckpt_path = os.path.join(data_dir, 'checkpints')

    test_data = GFKDtrack_dataset(data_dir, test_file)
    test_dataloader = DataLoader(test_data, batch_size=1, num_workers=4)
    model.load_state_dict(torch.load(os.path.join(ckpt_path, ckp_file)))

    print(f'Loading the {ckp_file} checkpoint file')
    pbar = enumerate(test_dataloader)
    pbar = tqdm(pbar)
    model.eval()
    for i, (imgs, labels) in pbar:
        imgs = imgs.to(device)
        # labels = labels.to(device)
        pred = model(imgs)
        ipdb.set_trace()
        p = np.array(pred.detach().cpu())   # shape = (batch_size, 2)
        # print()

if __name__ == "__main__":

    if len(sys.argv) == 2:
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test(data_dir, 'epoch_100.pt', 'test.json', device)


