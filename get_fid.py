import argparse
from distutils.util import strtobool
import numpy as np
from PIL import Image
import random
from scipy import linalg
import torch
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from dataset import FoodDataset
from inception import InceptionV3
from model import Generator, Discriminator

# 引用先: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py


def calculate_activation_statistics(
    images,
    model,
    dims = 2048,
    device = "cpu"
):
    model.eval()
    act=np.empty((len(images), dims))

    batch=images.to(device)
    pred = model(batch)[0]

    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    act= pred.cpu().data.numpy().reshape(pred.size(0), -1)
    
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_fretchet(images_real,images_fake,model, device):
     mu_1,std_1 = calculate_activation_statistics(images_real,model,device=device)
     mu_2,std_2 = calculate_activation_statistics(images_fake,model,device=device)
    
     """get fretched distance"""
     fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
     return fid_value


# Frechet Inception Distance(FID)を計算する
def get_fid(args, model):

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

    # シード固定
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    # 学習したモデルの読み込み
    netG = Generator(nz = args.z_dim).to(device)
    netG.load_state_dict(torch.load(args.netG_path, map_location=device))

    model = model.to(device)

    fid_list = []

    for itr, data in tqdm(enumerate(dataloader)):
        # 本物画像の取得
        if args.dataset == "mnist":
            real_image = data[0].to(device)
        elif args.dataset == "food":
            real_image = data.to(device)

        # 偽物画像の生成
        sample_size = real_image.size(0)
        noise = torch.randn(sample_size, args.z_dim, 1, 1, device=device)
        fake_image = netG(noise)

        # FIDの計算
        fid = calculate_fretchet(real_image, fake_image, model, device)
        fid_list.append(fid)

        if itr%20 == 0:
            print(f"Iteratoin: {itr}, FID: {fid}")

    print(f"FID: {np.mean(fid_list)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Compute FID for a trained gans model.')
    parser.add_argument('--dataset', type=str, default="mnist", choices=["mnist", "food"])
    parser.add_argument('--datadir', type=str, default="./data/food")
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--netG_path', type=str, 
        default="./result/20220730-185127_mnist_lsgan_64_0.0002_0_5_0.5/models/netG_epoch_100.pth")
    args = parser.parse_args()

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx])

    get_fid(args, model)
