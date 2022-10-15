import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from model import Generator, Discriminator, weights_init

# 引用先: http://cedro3.com/ai/pytorch-conditional-gan/


# 学習
def train(dataloader, args, writer=None):
    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device
    print('device:', device)

    # モデルの作成
    netG = Generator(nz = args.z_dim).to(device)   # 入力ベクトルの次元は、nz
    netG.apply(weights_init)
 
    netD = Discriminator(nch = args.n_channel, dropout=args.dropout).to(device)   # 入力Tensorのチャネル数は、nch
    netD.apply(weights_init)
 
    # loss関数の定義
    if args.loss == "bce":
        criterion = nn.BCELoss()
    elif args.loss == "lsgan":
        criterion = nn.MSELoss()
    else:
        print("Loss must be bce or lsgan.")
        return
 
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=1e-5)
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=1e-5)

    # 生成で使用するノイズ・特徴量
    fixed_noise = torch.randn(args.batch_size, args.z_dim, 1, 1, device=device)

    Loss_D_list, Loss_G_list = [], []  # グラフ作成用リスト初期化
    loop_i = 0

    # 生成画像の保存先
    img_dir = os.path.join(args.outf, "images")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir, exist_ok=True)
    
    # モデルの保存先
    model_dir = os.path.join(args.outf, "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    
    print("training start")
    
    # 学習
    for epoch in range(args.n_epoch):
        for itr, data in tqdm(enumerate(dataloader)):

            # 本物画像の取得
            if args.dataset == "mnist":
                real_image = data[0].to(device)
            elif args.dataset == "food":
                real_image = data.to(device)

            sample_size = real_image.size(0)
            noise = torch.randn(sample_size, args.z_dim, 1, 1, device=device)  # ランダムベクトル生成（ノイズ）

            # 正解ラベルにはノイズを含ませる
            _real_target = [random.uniform(0.7, 1.0) for _ in range(sample_size)]
            real_target = torch.tensor(_real_target, device=device)
            _fake_target = [random.uniform(0.0, 0.3) for _ in range(sample_size)]
            fake_target = torch.tensor(_fake_target, device=device)

            # -----識別器の更新------------------------------
            netD.zero_grad()

            # 本物のデータに対する識別器の予測
            output = netD(real_image)
            errD_real = criterion(output, real_target)
            D_x = output.mean().item()

            # 偽物画像の生成
            fake_image = netG(noise)

            # 偽物のデータに対する識別器の予測
            output = netD(fake_image.detach())
            errD_fake = criterion(output, fake_target)
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
            errD.backward()

            # 識別器の更新頻度を減らす
            if itr % args.d_skip==0:
                optimizerD.step()

            # -----生成器の更新------------------------------
            netG.zero_grad()

            output = netD(fake_image)
            errG = criterion(output, real_target)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # ログの出力
            if itr % args.display_interval == 0:
                    print('[{}/{}][{}/{}] Loss_D: {:.3f} Loss_G: {:.3f} D(x): {:.3f} D(G(z)): {:.3f}/{:.3f}'
                        .format(epoch + 1, args.n_epoch,
                                itr + 1, len(dataloader),
                                errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                
                    Loss_D_list.append(errD.item())  # Loss_Dデータ蓄積 (グラフ用)
                    Loss_G_list.append(errG.item())  # Loss_Gデータ蓄積 (グラフ用)
            
            if writer is not None:
                writer.add_scalar("loss_D", errD.item(), loop_i)
                writer.add_scalar("loss_G", errG.item(), loop_i)
                writer.add_scalar("D(x)", D_x, loop_i)
                writer.add_scalar("D_G_z1", D_G_z1, loop_i)
                writer.add_scalar("D_G_z2", D_G_z2, loop_i)
                if itr % args.display_interval == 0:
                    #f_img = make_grid(fake_image)
                    f_img = fake_image[0].squeeze(dim=0)
                    writer.add_image("fake_image", f_img, loop_i)
            
            # 本物画像の保存
            if epoch == 0 and itr == 0:
                save_image(real_image, '{}/real_samples.png'.format(args.outf),
                           normalize=True, nrow=8)
            
            loop_i += 1

        # 生成した画像の保存
        fake_image = netG(fixed_noise)
        save_image(fake_image.detach(), '{}/fake_samples_epoch_{:03d}.png'.format(img_dir, epoch+1),
                   normalize=True, nrow=8)
        
        # モデルの保存
        if (epoch + 1) % 10 == 0:  
            torch.save(netG.state_dict(), '{}/netG_epoch_{}.pth'.format(model_dir, epoch+1))
            torch.save(netD.state_dict(), '{}/netD_epoch_{}.pth'.format(model_dir, epoch+1))
