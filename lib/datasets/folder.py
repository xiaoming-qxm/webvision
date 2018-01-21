# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

"""Modified from original pytorch version. """

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def make_dataset_spec(root_dir):
    """special version."""
    images = []
    root_dir = os.path.abspath(root_dir)
    class_idx = int(root_dir.split("/")[-1])

    for fname in sorted(os.listdir(root_dir)):
        if is_image_file(fname):
            path = os.path.join(root_dir, fname)
            item = (path, class_idx)
            images.append(item)

    return images


def make_dataset_gen(root_dir):
    """generic version."""
    images = []
    root_dir = os.path.abspath(root_dir)
    num_classes = 0

    for cls_idx in sorted(os.listdir(root_dir)):
        d = os.path.join(root_dir, cls_idx)
        if not os.path.isdir(d):
            continue

        for fname in sorted(os.listdir(d)):
            if is_image_file(fname):
                path = os.path.join(root_dir, cls_idx, fname)
                item = (path, int(cls_idx))
                images.append(item)
        num_classes += 1

    return images, num_classes


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class SpecImageFolder(data.Dataset):
    """ A special data loader where the images are arranged in this way: ::

       root/001.jpg
       root/002.jpg
       root/003.jpg
       ...

       Note: root contains the label, see `__init__` for details.

       Attributes:
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        imgs = make_dataset_spec(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in folders of: "
                               + root + "\nSupported image extensions are: "
                               + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.class_idx = int(root.split("/")[-1])
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class GenImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/0/xxx.png
        root/0/xxy.png
        root/0/xxz.png

        root/1/123.png
        root/1/nsdf3.png
        root/1/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        imgs, num_classes = make_dataset_gen(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root +
                               "\nSupported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.num_classes = num_classes
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
