# Taken from https://github.com/shengliu66/ELR/blob/master/ELR_plus/data_loader/webvision.py
from PIL import Image
from torch.utils.data.dataset import Subset
import torch
from torchvision import transforms


def get_webvision(root, train=True, num_class=50):
    transform_train = transforms.Compose([
        transforms.RandomCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_val = transforms.Compose([
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_imagenet = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if train:
        train_dataset = Webvision(root, train=train, transform=transform_train, num_class=num_class)
        val_dataset = Webvision(root, val=train, transform=transform_val, num_class=num_class)
        print(f"Train: {len(train_dataset)} WebVision Val: {len(val_dataset)}")
    else:
        train_dataset = []
        val_dataset = ImagenetVal(root, transform=transform_imagenet, num_class=num_class)
        print(f"Imagnet Val: {len(val_dataset)}")

    return train_dataset, val_dataset


class ImagenetVal(torch.utils.data.Dataset):
    def __init__(self, root, transform, num_class):
        self.root = root + 'imagenet/'
        self.transform = transform

        with open(self.root + 'imagenet_val.txt') as f:
            lines = f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    self.val_imgs.append(img)
                    self.val_labels[img] = target

    def __getitem__(self, index):
        img_path = self.val_imgs[index]
        target = self.val_labels[img_path]
        image = Image.open(self.root + 'val/' + img_path).convert('RGB')
        img = self.transform(image)

        return img, target  # , index, target

    def __len__(self):
        return len(self.val_imgs)


class Webvision(torch.utils.data.Dataset):
    def __init__(self, root, train=False, val=False, test=False, transform=None,
                 num_class=50):
        self.root = root
        self.transform = transform
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}

        self.train = train
        self.val = val
        self.test = test

        if self.val:
            with open(self.root + 'info/val_filelist.txt') as f:
                lines = f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    self.val_imgs.append(img)
                    self.val_labels[img] = target
        elif self.test:
            with open(self.root + 'info/val_filelist.txt') as f:
                lines = f.readlines()
            self.test_imgs = []
            self.test_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    self.test_imgs.append(img)
                    self.test_labels[img] = target
        else:
            with open(self.root + 'info/train_filelist_google.txt') as f:
                lines = f.readlines()
            train_imgs = []
            self.train_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    train_imgs.append(img)
                    self.train_labels[img] = target

            self.train_imgs = train_imgs

    def __getitem__(self, index):
        if self.train:
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(self.root + img_path)
            img0 = image.convert('RGB')
            img0 = self.transform(img0)
            return img0, target  # , index, target
        elif self.val:
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]
            image = Image.open(self.root + 'val_images_256/' + img_path).convert('RGB')
            img = self.transform(image)
            return img, target  # , index, target
        elif self.test:
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]
            image = Image.open(self.root + 'val_images_256/' + img_path).convert('RGB')
            img = self.transform(image)
            return img, target  # , index, target

    def __len__(self):
        if self.test:
            return len(self.test_imgs)
        if self.val:
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)
