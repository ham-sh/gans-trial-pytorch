import glob
import os

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

from utils import load_heic


# 自分で撮影した画像用のデータセット
class FoodDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.load_img()
    
    def load_img(self):
        imgs_path = sorted(glob.glob(os.path.join(self.data_dir, "*")))
        self.imgs = []

        for path in tqdm(imgs_path):
            # 画像の読み込み
            if path[-4:] == "HEIC":
                try:
                    img = load_heic(path)
                except:
                    print(f"Couldn't load {path}.")
                    pass
            else:
                img = Image.open(path)
            
            img = img.resize((256, 256))   # リサイズ
            self.imgs.append(img)

        print(f"Loaded {len(self.imgs)} images.")
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img
