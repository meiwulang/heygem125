# decompyle3 version 3.9.2
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.19 (default, Mar 20 2024, 15:27:52) 
# [Clang 14.0.6 ]
# Embedded file name: /code/landmark2face_wy/data/image_folder.py
# Compiled at: 2024-03-06 17:51:22
# Size of source mod 2**32: 1893 bytes
"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""
import torch.utils.data as data
from PIL import Image
import os, os.path
IMG_EXTENSIONS = [
 ".jpg", ".JPG", ".jpeg", ".JPEG",
 ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP"]

def is_image_file(filename):
    return any((filename.endswith(extension) for extension in IMG_EXTENSIONS))


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir
    for (root, _, fnames) in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

        return images[:min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert("RGB")


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False, loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise RuntimeError("Found 0 images in: " + root + "\nSupported image extensions are: " + ",".join(IMG_EXTENSIONS))
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return (img, path)
        return img

    def __len__(self):
        return len(self.imgs)

