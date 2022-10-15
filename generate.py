import argparse
import numpy as np
import random
import torch

from datetime import datetime
from torchvision.utils import save_image

from model import Generator


# 学習したモデルを用いて画像生成を行う
def generate(args):
    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    # 学習したモデルの読み込み
    netG = Generator(nz = args.z_dim).to(device)
    netG.load_state_dict(torch.load(args.netG_path, map_location=device))

    noise = torch.randn(args.batch_size, args.z_dim, 1, 1, device=device)

    # 偽物画像の生成
    fake_image = netG(noise)

    save_image(fake_image.detach(), '{}/fake_samples_{}.png'.format(args.outf, datetime.now().strftime("%Y%m%d-%H%M%S")),
               normalize=True, nrow=8)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate fake images with a trained gans model.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--netG_path', type=str, 
        default="./result/20220730-185127_mnist_lsgan_64_0.0002_0_5_0.5/models/netG_epoch_100.pth")
    parser.add_argument('--outf', type=str, default='./result')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    generate(args)
