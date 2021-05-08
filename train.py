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
import torchvision
import torch.nn as nn

data_dir = '../data/mydata'

logger = logging.getLogger(__name__)


def train(data_dir, epochs, device):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyNet()
    # model = torchvision.models.resnet18(pretrained=True)
    # model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=7, stride=2, padding=3,
    #                            bias=False)
    # model.fc = nn.Linear(model.fc.in_features, 2)

    logging.info(f'{model}')
    model.to(device=device)
    ckpt_path = os.path.join(data_dir, 'checkpints')
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    train_data = GFKDtrack_dataset(data_dir, 'train.json')
    test_data = GFKDtrack_dataset(data_dir, 'test.json')
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=4, num_workers=4)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    start_epoch = 0
    logger.info(f'Total training epochs : {epochs}\n'
                f'Total training images : {len(train_data)}\n')
    test_flag = True

    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        optimizer.zero_grad()

        print(('%10s' * 3) % ('Epoch', 'loss', 'img_size'))
        epoch_loss = 0
        pbar = enumerate(train_dataloader)
        pbar = tqdm(pbar)
        for i, (imgs, labels) in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            # print()
            loss = criterion(pred, labels)
            # ipdb.set_trace()
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            test_flag = True

            s = ('%10s' + '%10.4g' * 2) % (
                '%g/%g' % (epoch, epochs - 1), loss, imgs.shape[-1])
            pbar.set_description(s)
        model.eval()

        logger.info(('%s:%5d   %s:%10.6f') % ('Epoch', epoch, 'epcoh loss', epoch_loss))

        if epoch%10==0:
            print(('%10s' * 3) % ('Test epoch', 'loss', 'img_size'))
            epoch_loss = 0
            pbar = enumerate(test_dataloader)
            pbar = tqdm(pbar)
            for i, (imgs, labels) in pbar:
                imgs = imgs.to(device)
                labels = labels.to(device)
                pred = model(imgs)
                loss = criterion(pred, labels)
                # ipdb.set_trace()
                epoch_loss += loss.item()
                s = ('%10s' + '%10.4g' * 2) % (
                    '%g/%g' % (epoch, epochs - 1), loss, imgs.shape[-1])
                pbar.set_description(s)
            logger.info(('----%s:%5d   %s:%10.6f') % ('Test epoch', epoch, 'epcoh loss', epoch_loss))
            # print()
        if epoch % 100 == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_path, f'epoch_{epoch}.pt'))
            logger.info(f'Saved the {epoch} epoch checkpoint!')


def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    log_dir: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    import datetime
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    # 此处不能使用logging输出
    print('log file path:' + log_file)

    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging

if __name__ == "__main__":

    if len(sys.argv) == 2:
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

    logging = init_logger(log_dir='log')


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    train(data_dir, 10000, device)
