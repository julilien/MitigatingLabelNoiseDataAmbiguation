# Taken from https://github.com/shengliu66/ELR/blob/master/ELR_plus/data_loader/clothing1m.py
from PIL import Image
from torch.utils.data.dataset import Subset
import torch
import random
from torchvision import transforms


def get_clothing(root, num_samples=0, train=True):
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
    ])

    if train:
        train_dataset = Clothing(root, num_samples=num_samples, train=train, transform=transform_train)
        val_dataset = Clothing(root, val=train, transform=transform_val)
        print(f"Train: {len(train_dataset)} Val: {len(val_dataset)}")

    else:
        train_dataset = []
        val_dataset = Clothing(root, test=(not train), transform=transform_val)
        print(f"Test: {len(val_dataset)}")

    return train_dataset, val_dataset


class Clothing(torch.utils.data.Dataset):

    def __init__(self, root, num_samples=0, train=False, val=False, test=False, transform=None,
                 num_class=14):
        self.root = root
        self.transform = transform
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}

        self.train = train
        self.val = val
        self.test = test

        with open('%s/noisy_label_kv.txt' % self.root, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = '%s/' % self.root + entry[0][7:]
                self.train_labels[img_path] = int(entry[1])
        with open('%s/clean_label_kv.txt' % self.root, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = '%s/' % self.root + entry[0][7:]
                self.test_labels[img_path] = int(entry[1])

        if train:
            train_imgs = []
            with open('%s/noisy_train_key_list.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for i, l in enumerate(lines):
                    img_path = '%s/' % self.root + l[7:]
                    train_imgs.append((i, img_path))
            self.num_raw_example = len(train_imgs)
            random.shuffle(train_imgs)
            class_num = torch.zeros(num_class)
            self.train_imgs = []
            for id_raw, impath in train_imgs:
                label = self.train_labels[impath]
                if class_num[label] < (num_samples / 14) and len(self.train_imgs) < num_samples:
                    self.train_imgs.append((id_raw, impath))
                    class_num[label] += 1
            random.shuffle(self.train_imgs)

        elif test:
            self.test_imgs = []
            with open('%s/clean_test_key_list.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/' % self.root + l[7:]
                    self.test_imgs.append(img_path)
        elif val:
            self.val_imgs = []
            with open('%s/clean_val_key_list.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/' % self.root + l[7:]
                    self.val_imgs.append(img_path)

    def __getitem__(self, index):
        if self.train:
            id_raw, img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
        elif self.val:
            img_path = self.val_imgs[index]
            target = self.test_labels[img_path]
        elif self.test:
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]
        image = Image.open(img_path).convert('RGB')
        if self.train:
            img0 = self.transform(image)

        if self.test or self.val:
            img = self.transform(image)
            return img, target
        else:
            return img0, target

    def __len__(self):
        if self.test:
            return len(self.test_imgs)
        if self.val:
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)

    def flist_reader(self, flist):
        imlist = []
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                row = line.split(" ")
                impath = self.root + row[0]
                imlabel = float(row[1].replace('\n', ''))
                imlist.append((impath, int(imlabel)))
        return imlist
