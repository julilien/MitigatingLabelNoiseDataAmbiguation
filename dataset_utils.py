import numpy as np
import os

from numpy.testing import assert_array_almost_equal
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

_TRAIN_INDICES_PATH = 'train_indices.npy'
_VAL_INDICES_PATH = 'val_indices.npy'

MNIST_STR = "mnist"
FASHION_MNIST_STR = "fashionmnist"
CIFAR10_STR = "cifar10"
CIFAR100_STR = "cifar100"
SVHN_STR = "svhn"
STL10_STR = "stl10"
IMAGENET_STR = "imagenet"
CIFAR10N_CLEAN_STR = "cifar10n_clean"
CIFAR10N_WORSE_STR = "cifar10n_worse"
CIFAR10N_AGGRE_STR = "cifar10n_aggre"
CIFAR10N_RANDOM1_STR = "cifar10n_random1"
CIFAR10N_RANDOM2_STR = "cifar10n_random2"
CIFAR10N_RANDOM3_STR = "cifar10n_random3"
CIFAR100N_CLEAN_STR = "cifar100n_clean"
CIFAR100N_NOISY_STR = "cifar100n_noisy"
CLOTHING_1M_STR = "clothing1m"
WEBVISION_STR = "webvision"


def split_train_and_val_data(raw_trainset, args, shuffle, num_workers=4, pin_memory=True):
    ds_size = len(raw_trainset)
    indices = list(range(ds_size))
    split = int(np.floor(args.val_split_prop * ds_size))

    full_train_indices_path = os.path.join(args.save_path, _TRAIN_INDICES_PATH)
    full_val_indices_path = os.path.join(args.save_path, _VAL_INDICES_PATH)

    if os.path.exists(full_train_indices_path) and os.path.exists(full_val_indices_path):
        train_indices = np.load(full_train_indices_path)
        val_indices = np.load(full_val_indices_path)
    else:
        # Shuffle indices
        if shuffle:
            np.random.seed(args.seed)
            np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]

        # Save generated indices
        np.save(full_train_indices_path, train_indices)
        np.save(full_val_indices_path, val_indices)

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    trainloader = DataLoader(raw_trainset, batch_size=args.batch_size, sampler=train_sampler, num_workers=num_workers,
                             pin_memory=pin_memory)
    valloader = DataLoader(raw_trainset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=num_workers,
                           pin_memory=pin_memory)

    return trainloader, valloader


def get_cifarn_random_labels(file, dataset_str):
    if dataset_str in [CIFAR10N_CLEAN_STR, CIFAR100N_CLEAN_STR]:
        label_id = "clean_label"
    elif dataset_str == CIFAR10N_WORSE_STR:
        label_id = "worse_label"
    elif dataset_str == CIFAR10N_AGGRE_STR:
        label_id = "aggre_label"
    elif dataset_str == CIFAR10N_RANDOM1_STR:
        label_id = "random_label1"
    elif dataset_str == CIFAR10N_RANDOM2_STR:
        label_id = "random_label2"
    elif dataset_str == CIFAR10N_RANDOM3_STR:
        label_id = "random_label3"
    elif dataset_str == CIFAR100N_NOISY_STR:
        label_id = "noisy_label"
    else:
        raise ValueError("Unknown key CIFAR-10[0]N variant '{}'.".format(dataset_str))
    return file[label_id]


def get_asymmetric_noise_for_dataset(dataset, label_noise, num_classes, training_targets, seed):
    assert dataset in [CIFAR10_STR, CIFAR100_STR], "Currently, asymmetric noise is only supported for " \
                                                   "CIFAR-10 and CIFAR-100."

    training_targets = training_targets.numpy()

    # Set NumPy random seed
    np.random.seed(seed)

    if dataset == CIFAR10_STR:
        for i in range(num_classes):
            indices = np.where(training_targets == i)[0]
            np.random.shuffle(indices)
            for j, idx in enumerate(indices):
                if j < label_noise * len(indices):
                    # truck -> automobile
                    if i == 9:
                        training_targets[idx] = 1
                    # bird -> airplane
                    elif i == 2:
                        training_targets[idx] = 0
                    # cat -> dog
                    elif i == 3:
                        training_targets[idx] = 5
                    # dog -> cat
                    elif i == 5:
                        training_targets[idx] = 3
                    # deer -> horse
                    elif i == 4:
                        training_targets[idx] = 7
    elif dataset == CIFAR100_STR:
        # Code taken from the SOP repository
        def multiclass_noisify(y, P, random_state=0):
            """ Flip classes according to transition probability matrix T.
            It expects a number between 0 and the number of classes - 1.
            """

            assert P.shape[0] == P.shape[1]
            assert np.max(y) < P.shape[0]

            # row stochastic matrix
            assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
            assert (P >= 0.0).all()

            m = y.shape[0]
            new_y = y.copy()
            flipper = np.random.RandomState(random_state)

            for idx in np.arange(m):
                i = y[idx]
                # draw a vector with only an 1
                flipped = flipper.multinomial(1, P[i, :], 1)[0]
                new_y[idx] = np.where(flipped == 1)[0]

            return new_y

        def build_for_cifar100(size, noise):
            """ random flip between two random classes.
            """
            assert (noise >= 0.) and (noise <= 1.)

            P = np.eye(size)
            cls1, cls2 = np.random.choice(range(size), size=2, replace=False)
            P[cls1, cls2] = noise
            P[cls2, cls1] = noise
            P[cls1, cls1] = 1.0 - noise
            P[cls2, cls2] = 1.0 - noise

            assert_array_almost_equal(P.sum(axis=1), 1, 1)
            return P

        P = np.eye(num_classes)
        n = label_noise
        nb_superclasses = 20
        nb_subclasses = 5

        if n > 0.0:
            for i in np.arange(nb_superclasses):
                init, end = i * nb_subclasses, (i + 1) * nb_subclasses
                P[init:end, init:end] = build_for_cifar100(nb_subclasses, n)

            y_train_noisy = multiclass_noisify(training_targets, P=P,
                                               random_state=seed)
            actual_noise = (y_train_noisy != training_targets).mean()
            assert actual_noise > 0.0
            training_targets = y_train_noisy
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

    return torch.tensor(training_targets, dtype=torch.int64)
