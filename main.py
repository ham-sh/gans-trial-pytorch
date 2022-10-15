import argparse
import numpy as np
import os
import random
import torch

from datetime import datetime
from distutils.util import strtobool
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataset import FoodDataset
from train import train

def main(args):

    # データセットの取得
    if args.dataset == "mnist":
        transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(args.img_size),
            transforms.ToTensor()
        ])
        dataset = datasets.MNIST(
            "./data",
            train = True,
            download = True,
            transform = transform
        )
    
    elif args.dataset == "food":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=0.2, scale=(0.8,1.2)),
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            #transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
        ])
        dataset = FoodDataset(args.datadir, transform)

    # データローダの取得
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=int(args.workers)
    )

    args.outf = args.outf + "/" + datetime.now().strftime("%Y%m%d-%H%M%S")\
                            + "_" + str(args.dataset) +"_" + str(args.loss) + "_" + str(args.batch_size) + "_" + str(args.lr)\
                            + "_" + str(args.seed) + "_" + str(args.d_skip) + "_" + str(args.dropout)
    logdir = args.outf + "/logs/"

    if not args.debug:
        if not os.path.exists(args.outf):
            os.makedirs(args.outf, exist_ok=True)

        print("Log is saved at {}.".format(logdir))
        writer = SummaryWriter(logdir)

    # デバッグ時にはNoneにする
    else:
        writer = None

    # シード固定
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 学習
    train(dataloader, args, writer)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='GANs trial.')
    parser.add_argument('--display_interval', type=int, default=100)
    parser.add_argument('--dataset', type=str, default="mnist", choices=["mnist", "food"])
    parser.add_argument('--datadir', type=str, default="./data/food")
    parser.add_argument('--loss', type=str, default="lsgan", choices=["bce", "lsgan"])
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--n_channel', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--n_epoch', type=int, default=1000)
    parser.add_argument('--d_skip', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--outf', type=str, default='./result')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--debug', type=strtobool, default=False)
    args = parser.parse_args()

    main(args)
