# gans-trial-pytorch

GANsを実装してみました。

## How to use
### 学習
```
python main.py
```
* 主なオプション
    * `--dataset`: 使用するデータセット。`mnist`か`food`。今回、自分で撮り溜めた約600枚のご飯の画像を使用しました (公開はしていないので適宜学習させたい画像を使用してください) 。
    * `--datadir`: 学習に使用するデータのパス。MNISTを使う場合は指定不要。
    * `--loss`: Binary Cross Entropy Loss(`bce`)とLeast squares generative adversarial networks (`lsgans`)を実装。
    * `--z_dim`: Generatorに入力するノイズの次元。
    * `--dropout`: Discriminatorの最終層に挿入するDropoutの割合。
    * `--d_skip`: Discriminatorの重み更新1回に対してGeneratorの重みを何回更新させるか。
    * `--outf`: モデル、生成画像、tensorboardのログを含む学習結果の保存先。
* Command examples.
    ```
    python main.py --workers 0 --dataset food --datadir ./data/food/ --d_skip 10 --outf ./result --seed 0 --device cuda:0
    ```
* Tensorboardでのログの見方
    ```
    tensorboard --logdir {PATH TO THE LOG}
    ```

<br>

### Frechet Inception Distance (FID) の計算
モデルの定量評価のためにFIDの計算を実装しました。
inception.pyはFIDの計算で使用しています。
```
python get_fid.py
```
* 主なオプション
    * `--dataset`: 使用するデータセット。
    * `--netG_path`: 学習済みモデルの保存先
* Example:
    ```
    python get_fid.py --dataset mnist --netG_path ./data/result/20220731-113841_mnist_lsgan_64_0.0002_1041_1_0.5/models/netG_epoch_3000.pth
    ```

<br>

### 画像生成
学習したモデルを用いて画像生成のみ行う場合は以下を実行します。
```
python generate.py
```
* 主なオプション
    * `--netG_path`: 学習済みモデルの保存先
* Example:
    ```
    python generate.py --netG_path ./data/result/20220731-113841_mnist_lsgan_64_0.0002_1041_1_0.5/models/netG_epoch_3000.pth --outf ./result
    ```