from typing import Tuple, Any, Optional, Callable

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, CIFAR100, SVHN, STL10, VisionDataset
import torchvision.transforms as transforms
import logging
from PIL import Image
from pathlib import Path

from data.augmentation_archive import fa_reduced_cifar10, autoaug_paper_cifar10, autoaug_policy
from data.augmentations import Augmentation, CutoutDefault
from data.clothing1m import get_clothing
from data.webvision import get_webvision
from dataset_utils import *

TRAIN_TARGETS_FN = "train_targets.pt"
TRAIN_ORIG_TARGETS_FN = "train_original_targets.pt"
TRAIN_DATA_FN = "train_data.pt"
TRAIN_ORIG_DATA_FN = "train_original_data.pt"
SELECTED_LABELS_FN = "selected_labels.pt"

FADING_IN_OPACITIES_NP_FN = "fading_in_opacities.npy"

TEST_DATA_FN = "test_data.pt"
TEST_TARGETS_FN = "test_targets.pt"

TRAIN_ACTIVATIONS_NP_FN = "train_activations.npy"
TRAIN_LABELS_NP_FN = "train_labels.npy"
TEST_ACTIVATIONS_NP_FN = "test_activations.npy"
TEST_LABELS_NP_FN = "test_labels.npy"

# Dataset-specific means and stddevs
IMAGENET_MEAN = [0.4914, 0.4822, 0.4465]
IMAGENET_STDDEV = [0.2023, 0.1994, 0.2010]
MNIST_MEAN = (0.1307,)
MNIST_STDDEV = (0.3081,)


def load_dataset(dataset_name, data_dir, custom=False, data_aug=False, cr_aug: Optional[str] = None,
                 cutout: Optional[int] = None, sample_fraction=None, seed=0):
    if dataset_name == CIFAR10_STR:
        logging.debug('Dataset: CIFAR10.')

        if data_aug:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDDEV),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDDEV),
            ])
        trainset = CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        testset = CIFAR10(root=data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDDEV),
        ]))

        num_classes = 10
    elif dataset_name == SVHN_STR:
        logging.debug('Dataset: SVHN.')

        if data_aug:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDDEV),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDDEV),
            ])
        trainset = CustomSVHN(root=data_dir, split="train", download=True, transform=train_transform)

        testset = CustomSVHN(root=data_dir, split="test", download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDDEV),
        ]))
        num_classes = 10
    elif dataset_name == CIFAR100_STR:
        logging.debug('Dataset: CIFAR-100.')

        if data_aug:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDDEV),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDDEV),
            ])
        trainset = CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)

        testset = CIFAR100(root=data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDDEV),
        ]))

        num_classes = 100
    elif dataset_name == FASHION_MNIST_STR:
        logging.debug('Dataset: Fashion-MNIST.')
        if not custom:
            trainset = FashionMNIST(root=data_dir, train=True, download=True, transform=transforms.Compose([
                transforms.Grayscale(3), transforms.Resize(32), transforms.ToTensor(),
                transforms.Normalize(MNIST_MEAN, MNIST_STDDEV)
            ]))
        else:
            trainset = CustomFashionMNIST(root=data_dir, train=True, download=True, transform=transforms.Compose([
                transforms.Grayscale(3), transforms.Resize(32), transforms.ToTensor(),
                transforms.Normalize(MNIST_MEAN, MNIST_STDDEV)
            ]))

        testset = FashionMNIST(root=data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.Grayscale(3), transforms.Resize(32), transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STDDEV)
        ]))
        num_classes = 10
    elif dataset_name == MNIST_STR:
        logging.debug('Dataset: MNIST.')
        if not custom:
            trainset = MNIST(root=data_dir, train=True, download=True, transform=transforms.Compose([
                transforms.Grayscale(3), transforms.Resize(32), transforms.ToTensor(),
                transforms.Normalize(MNIST_MEAN, MNIST_STDDEV)
            ]))
        else:
            trainset = CustomMNIST(root=data_dir, train=True, download=True, transform=transforms.Compose([
                transforms.Grayscale(3), transforms.Resize(32), transforms.ToTensor(),
                transforms.Normalize(MNIST_MEAN, MNIST_STDDEV)
            ]))

        testset = MNIST(root=data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.Grayscale(3), transforms.Resize(32), transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STDDEV)
        ]))
        num_classes = 10
    elif dataset_name == STL10_STR:
        logging.info("Dataset: STL10")

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDDEV),
        ])

        trainset = CustomSTL10(root=data_dir, split='train', download=True, transform=train_transform)

        testset = CustomSTL10(root=data_dir, split='test', download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDDEV),
        ]))
        num_classes = 10
    elif dataset_name in [CIFAR10N_CLEAN_STR, CIFAR10N_WORSE_STR, CIFAR10N_AGGRE_STR, CIFAR10N_RANDOM1_STR,
                          CIFAR10N_RANDOM2_STR, CIFAR10N_RANDOM3_STR]:
        logging.debug('Dataset: CIFAR10N (var: {}).'.format(dataset_name))

        if data_aug:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDDEV),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDDEV),
            ])
        trainset = CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)

        testset = CIFAR10(root=data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDDEV),
        ]))

        raw_target_data = torch.load("data/CIFAR-10_human.pt")
        trainset.targets = get_cifarn_random_labels(raw_target_data, dataset_name)

        num_classes = 10
    elif dataset_name in [CIFAR100N_CLEAN_STR, CIFAR100N_NOISY_STR]:
        logging.debug('Dataset: CIFAR-100N (var: {}).'.format(dataset_name))

        if data_aug:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDDEV),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDDEV),
            ])
        trainset = CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)

        testset = CIFAR100(root=data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDDEV),
        ]))

        raw_target_data = torch.load("data/CIFAR-100_human.pt")
        trainset.targets = get_cifarn_random_labels(raw_target_data, dataset_name)

        num_classes = 100
    else:
        raise ValueError("Unsupported dataset {}.".format(dataset_name))

    trainset.targets = torch.tensor(trainset.targets)
    testset.targets = torch.tensor(testset.targets)

    # Subsample
    if sample_fraction is not None:
        np_targets = trainset.targets.numpy()

        # Generate samples
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=(1. - sample_fraction), random_state=seed)
        sample_idxs, _ = next(sss.split(np.zeros_like(np_targets), np_targets))

        # Apply subsampling
        trainset.data = trainset.data[sample_idxs]
        trainset.targets = trainset.targets[sample_idxs]


    if cr_aug is not None and cr_aug != "":
        transform_train_aug = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDDEV),
        ])

        autoaug = transforms.Compose([])
        if cr_aug == 'fa_reduced_cifar10':
            autoaug.transforms.insert(0, Augmentation(fa_reduced_cifar10()))
        elif cr_aug == 'autoaug_cifar10':
            autoaug.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
        elif cr_aug == 'autoaug_extend':
            autoaug.transforms.insert(0, Augmentation(autoaug_policy()))
        else:
            raise ValueError('not found augmentations. %s' % cr_aug)
        transform_train_aug.transforms.insert(0, autoaug)

        if cutout is not None and cutout > 0:
            transform_train_aug.transforms.append(CutoutDefault(cutout))

        trainset = AugmentedDataset(trainset, transform_train_aug)

    return trainset, testset, num_classes


def make_selected_dataset(args, val_split_prop=None, label_noise=0.0, selected_labels=None, shuffle_val_data=False,
                          num_workers=4, pin_memory=True, use_asymmetric_noise=False, cr_aug: Optional[str] = None,
                          cutout: Optional[int] = None):
    assert selected_labels is not None, "Selected labels must be provided as a list of two integers."

    if args.dataset == CLOTHING_1M_STR:
        trainset, valset = get_clothing(args.data_dir, train=True, num_samples=np.inf)
        _, testset = get_clothing(args.data_dir, train=False)

        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
                                 pin_memory=pin_memory)
        valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers,
                               pin_memory=pin_memory)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers,
                                pin_memory=pin_memory)

        num_classes = 14
        return trainloader, valloader, testloader, num_classes
    elif args.dataset == WEBVISION_STR:
        trainset, testset = get_webvision(args.data_dir, train=True)

        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
                                 pin_memory=pin_memory)
        valloader = None
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers,
                                pin_memory=pin_memory)

        num_classes = 50
        return trainloader, valloader, testloader, num_classes
    elif args.dataset == IMAGENET_STR:
        logging.warning("Currently, only the evaluation on ImageNet is supported.")
        _, testset = get_webvision(args.data_dir, train=False)

        trainloader = None
        valloader = None
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers,
                                pin_memory=pin_memory)

        num_classes = 50
        return trainloader, valloader, testloader, num_classes

    trainset, testset, num_classes = load_dataset(args.dataset, args.data_dir, data_aug=args.data_aug, cr_aug=cr_aug,
                                                  cutout=cutout, sample_fraction=args.sample_fraction, seed=args.seed)

    if not (args.dataset.lower().startswith("cifar10n") or args.dataset.lower().startswith("cifar100n")):
        # Selected classes
        selected_indices = sum(trainset.targets == i for i in selected_labels).bool()
        trainset.targets = trainset.targets[selected_indices]
        trainset.data = trainset.data[selected_indices]

        selected_indices_test = sum(testset.targets == i for i in selected_labels).bool()
        testset.targets = testset.targets[selected_indices_test]
        testset.data = testset.data[selected_indices_test]

        # Flip classes by chance
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

        logging.debug("Label noise: {}".format(label_noise))
        logging.debug("Targets before: {}".format(trainset.targets))

        torch.manual_seed(args.seed)

        if isinstance(label_noise, np.ndarray):
            # Flip labels
            label_noise_t = torch.tensor(label_noise)
            new_targets = torch.where(torch.rand(trainset.targets.size()) < label_noise_t[trainset.targets],
                                      torch.randint(0, len(selected_labels), trainset.targets.size()),
                                      trainset.targets)

            # Subsample instances to balance (aka stratifying)
            min_class_instances = torch.min(torch.bincount(new_targets))
            retained_labels = []
            for i in selected_labels:
                indices = (new_targets == i).nonzero().squeeze().tolist()
                if not type(indices) == list:
                    indices = list(indices)
                indices = indices[:min_class_instances]
                retained_labels += indices

            torch.save(trainset.targets[retained_labels], os.path.join(args.save_path, TRAIN_ORIG_TARGETS_FN))
            trainset.targets = new_targets

            trainset.targets = trainset.targets[retained_labels]
            trainset.data = trainset.data[retained_labels]

            num_classes = len(selected_labels)
        else:
            torch.save(trainset.targets, os.path.join(args.save_path, TRAIN_ORIG_TARGETS_FN))

            if not use_asymmetric_noise:
                trainset.targets = torch.where(torch.rand(trainset.targets.size()) < label_noise,
                                               torch.randint(0, len(selected_labels), trainset.targets.size()),
                                               trainset.targets)
            else:
                assert args.dataset in ["cifar10", "cifar100"], "Currently, asymmetric noise is only supported for " \
                                                                "CIFAR-10 and CIFAR-100."

                trainset.targets = get_asymmetric_noise_for_dataset(args.dataset, label_noise, num_classes,
                                                                    trainset.targets, args.seed)
            num_classes = len(selected_labels)

        logging.debug("Targets after: {}".format(trainset.targets))
    else:
        torch.save(trainset.targets, os.path.join(args.save_path, TRAIN_ORIG_TARGETS_FN))

    # Save targets
    torch.save(trainset.targets, os.path.join(args.save_path, TRAIN_TARGETS_FN))
    torch.save(trainset.data, os.path.join(args.save_path, TRAIN_DATA_FN))
    torch.save(selected_labels, os.path.join(args.save_path, SELECTED_LABELS_FN))

    validation_loader = None

    if val_split_prop is not None:
        trainloader, validation_loader = split_train_and_val_data(trainset, args, shuffle=shuffle_val_data,
                                                                  num_workers=num_workers, pin_memory=pin_memory)
    else:
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
                                 pin_memory=pin_memory)

    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory)

    return trainloader, validation_loader, testloader, num_classes


def construct_preloaded_dataset(args, train_data, train_targets, selected_labels=None,
                                shuffle_val_data=False, four_class_problem=False, num_workers=4):
    assert selected_labels is not None, "Selected labels must be provided as a list of two integers."

    trainset, testset, _ = load_dataset(args.dataset, args.data_dir, data_aug=args.data_aug, custom=True,
                                        seed=args.seed)

    trainset.data = train_data
    trainset.targets = train_targets

    if four_class_problem:
        num_classes = torch.max(trainset.targets).cpu().numpy().item() + 1
    elif selected_labels is not None:
        num_classes = len(selected_labels)
    else:
        logging.warning("As the selected labels are not properly specified, the number of classes can not be "
                        "determined precisely.")
        num_classes = 2

    valloader = None
    if args is not None and args.val_split_prop is not None and args.val_split_prop > 0.0:
        trainloader, valloader = split_train_and_val_data(trainset, args, shuffle=shuffle_val_data)
    else:
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)

    selected_indices_test = sum(testset.targets == i for i in selected_labels).bool()
    testset.targets = testset.targets[selected_indices_test]
    testset.data = testset.data[selected_indices_test]

    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, valloader, testloader, num_classes


class CustomMNIST(MNIST):
    """
    The original MNIST Vision Dataset object only allows for single integer labels.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(CustomMNIST, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CustomFashionMNIST(FashionMNIST):
    """
    The original FashionMNIST Vision Dataset object only allows for single integer labels.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(CustomFashionMNIST, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CustomSVHN(SVHN):
    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(root, split, transform, target_transform, download)
        self.targets = torch.Tensor(self.labels).long()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CustomSTL10(STL10):
    def __init__(
            self,
            root: str,
            split: str = "train",
            folds: Optional[int] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(root=root, split=split, folds=folds, transform=transform, target_transform=target_transform,
                         download=download)
        self.targets = torch.Tensor(self.labels).long()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def make_reproducible_dataset(args, save_path, val_split_prop=None, label_noise=0.0, eval=False, subselect_classes=None,
                              shuffle_val_data=False, num_workers=4):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    trainset, testset, num_classes = load_dataset(args.dataset, args.data_dir, data_aug=args.data_aug,
                                                  sample_fraction=args.sample_fraction, seed=args.seed)

    validation_loader = None

    if label_noise > 0.0:
        # Flipping
        if not eval:
            logging.debug("Targets before: {}".format(trainset.targets))
            torch.save(trainset.targets, save_path + "/train_original_targets.pt")
            if torch.is_tensor(trainset.targets):
                trgt_size = trainset.targets.size()
                trgt_size2 = trgt_size
                old_trgts = trainset.targets

                trainset.targets = torch.where(torch.rand(trgt_size) < label_noise,
                                               torch.randint(0, num_classes, trgt_size2),
                                               old_trgts)
            else:
                trgt_size = len(trainset.targets)

                trainset.targets = np.where(np.random.random(trgt_size) < label_noise,
                                            np.random.randint(0, num_classes, trgt_size),
                                            trainset.targets)

            logging.debug("Targets after: {}".format(trainset.targets))

            torch.save(trainset.targets, os.path.join(save_path, TRAIN_TARGETS_FN))
            torch.save(trainset.data, os.path.join(save_path, TRAIN_DATA_FN))
        else:
            trainset.targets = torch.load(os.path.join(save_path, TRAIN_TARGETS_FN))
            logging.debug("Targets: {}".format(trainset.targets))
            trainset.data = torch.load(os.path.join(save_path, TRAIN_DATA_FN))

    if val_split_prop is not None:
        assert label_noise == 0.0, "No noise label in validation data"

        trainloader, validation_loader = split_train_and_val_data(trainset, args, shuffle=shuffle_val_data)
    else:
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
                                 worker_init_fn=lambda id: np.random.seed(id))

    if subselect_classes is not None:
        selected_indices_test = sum(testset.targets == i for i in subselect_classes).bool()
        testset.targets = testset.targets[selected_indices_test]
        testset.data = testset.data[selected_indices_test]

    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, validation_loader, testloader, num_classes


def subsample_data(dataset, val_split_prop, num_classes, sample_size, batch_size, num_workers=4):
    if val_split_prop is not None:
        raise NotImplementedError("val_split_prop not yet implemented for subset sample size.")

    total_sample_size = num_classes * sample_size
    cnt_dict = dict()
    total_cnt = 0
    indices = []
    for i in range(len(dataset)):

        if total_cnt == total_sample_size:
            break

        label = dataset[i][1]
        if label not in cnt_dict:
            cnt_dict[label] = 1
            total_cnt += 1
            indices.append(i)
        else:
            if cnt_dict[label] == sample_size:
                continue
            else:
                cnt_dict[label] += 1
                total_cnt += 1
                indices.append(i)

    indices = torch.tensor(indices)
    return DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(indices), num_workers=num_workers)


class AugmentedDataset(VisionDataset):
    def __init__(self, base_dataset, aug_transform):
        super().__init__(base_dataset.root, base_dataset.transform, base_dataset.target_transform)
        self.base_dataset = base_dataset
        self.aug_transform = aug_transform

    @property
    def targets(self):
        return self.base_dataset.targets

    @targets.setter
    def targets(self, value):
        self.base_dataset.targets = value

    @property
    def data(self):
        return self.base_dataset.data

    @data.setter
    def data(self, value):
        self.base_dataset.data = value

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        raw_img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        raw_img = Image.fromarray(raw_img)

        if self.base_dataset.transform is not None:
            img = self.base_dataset.transform(raw_img)
        else:
            img = raw_img

        img2 = None
        if self.aug_transform is not None:
            img2 = self.aug_transform(raw_img)

        if self.base_dataset.target_transform is not None:
            target = self.base_dataset.target_transform(target)

        return (img, img2), target


class SimpleTensorDataset(Dataset):
    def __init__(self, data, orig_targets, act_targets, transform=None):
        super().__init__()
        self.data = data
        self.orig_targets = orig_targets
        self.act_targets = act_targets

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raw_img = self.data[index]
        y = self.act_targets[index]
        orig_y = self.orig_targets[index]

        if not isinstance(raw_img, np.ndarray):
            raw_img = raw_img.numpy()

        x = Image.fromarray(raw_img)

        if self.transform:
            x = self.transform(x)

        return x, orig_y, y
