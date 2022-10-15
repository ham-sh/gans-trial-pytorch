import pyheif

from PIL import Image


# HEIC形式の画像の読み込み
def load_heic(file_name):
    heif_file = pyheif.read(file_name)
    img = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    return img
