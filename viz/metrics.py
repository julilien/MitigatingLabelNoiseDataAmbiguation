import numpy as np
import logging
import torch

from datasets import AugmentedDataset
from utils.utils import compute_binary_accuracy, compute_accuracy

CLUSTER_VAR_TAG = "variances"
CLUSTER_STDDEV_TAG = "stddevs"
# Actual cluster means (including corrupted instances)
CLUSTER_MEAN_TAG = "means"
CLUSTER_UNCORRUPTED_STDDEV_TAG = "uncorrupted_stddevs"  # (old: eucl_std_devs_uncorrupted)
CLUSTER_UNCORRUPTED_MEAN_TAG = "uncorrupted_means"  # (old: undistorted_centroids)
CLUSTER_CORRUPTED_MEAN_TAG = "corrupted_means"  # (old: distorted_centroids)
CORRUPTION_BIASES_TAG = "corruption_biases"


def prepare_features_and_labels(model, trainloader, args, fc_features):
    instances = []
    labels = []

    multi_input = isinstance(trainloader.dataset, AugmentedDataset)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # Targets store potentially corrupted and original label
        if multi_input:
            inputs = inputs[0]
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        model.eval()

        with torch.no_grad():
            model_outputs = model(inputs)

        features = fc_features.outputs[0][0]
        fc_features.clear()

        instances.append(features)
        labels.append(targets)

    return instances, labels


def get_class_means(instances, labels, classes, num_features=2):
    class_means = np.zeros([len(classes), num_features])

    for class_idx in classes:
        if len(np.shape(labels)) >= 2:
            mask = labels[:, 0] == class_idx
        else:
            mask = labels == class_idx

        selected_instances = instances[mask]
        class_means[class_idx] = np.mean(selected_instances, axis=0)

    return class_means


def get_class_variances(instances, labels, classes, class_means):
    # Statistics based on Euclidean distances (reduction applied)
    variances = np.zeros(len(classes))
    std_devs = np.zeros(len(classes))

    for class_idx in classes:
        if len(np.shape(labels)) >= 2:
            mask = labels[:, 0] == class_idx
        else:
            mask = labels == class_idx

        selected_instances = instances[mask]
        class_mean = class_means[class_idx]

        variances[class_idx] = np.var(np.linalg.norm(selected_instances - class_mean, axis=-1))
        std_devs[class_idx] = np.std(np.linalg.norm(selected_instances - class_mean, axis=-1))

    return variances, std_devs


def get_sigma_W(instances, labels, classes, class_means):
    Sigma_W = np.zeros(2)

    K = len(classes)
    for i in range(Sigma_W.shape[0]):
        n = 1
        for class_idx in classes:
            selected_instances = instances[labels[:, 0] == class_idx][:, i]
            n = selected_instances.shape[0]
            Sigma_W[i] += (selected_instances - class_means[class_idx, i]) @ np.transpose(
                selected_instances - class_means[class_idx, i])
        Sigma_W[i] /= K * n
    return Sigma_W


def get_centroids_by_corruption(args, instances, labels, classes, num_features=2):
    uncorrupted_centroids = np.zeros([len(classes), num_features])

    corrupted_mask = get_corrupted_instance_mask(labels)
    uncorrupted_mask = np.logical_not(corrupted_mask)

    for class_idx in classes:
        uncorrupted_instances = instances[uncorrupted_mask]
        uncorrupted_centroids[class_idx] = np.mean(uncorrupted_instances[labels[uncorrupted_mask][:, 0] == class_idx],
                                                   axis=0)

    corrupted_centroids = np.zeros([len(classes), len(classes), num_features])
    for class_idx in classes:
        for inner_class_idx in classes:
            # The second label dimension indicates the original class id
            corrupted_class_instances = instances[corrupted_mask][
                np.logical_and(labels[corrupted_mask][:, 0] == class_idx,
                               labels[corrupted_mask][:, 1] == inner_class_idx)]
            corrupted_centroids[class_idx, inner_class_idx] = np.mean(corrupted_class_instances, axis=0)

    return uncorrupted_centroids, corrupted_centroids


def get_uncorrupted_stddevs(args, instances, labels, uncorrupted_centroids, classes):
    """
    Calculates the standard deviation of the uncorrupted instances.

    :param instances:
    :param labels:
    :param uncorrupted_centroids:
    :param classes:
    :return:
    """

    uncorrupted_stddevs = np.zeros(len(classes))

    for class_idx in classes:
        uncorrupted_mask = np.logical_not(get_corrupted_instance_mask(labels))
        uncorrupted_instances = instances[uncorrupted_mask][labels[uncorrupted_mask, 0] == class_idx]
        uncorrupted_stddevs[class_idx] = np.std(
            np.linalg.norm(uncorrupted_instances - uncorrupted_centroids[class_idx], axis=-1))

    return uncorrupted_stddevs


def get_corrupted_instance_mask(labels):
    return labels[:, 0] != labels[:, 1]


def get_dot_product_bias(instances, labels, classes, class_means):
    corruption_biases = np.zeros([len(classes), len(classes) - 1])
    for class_idx in classes:
        class_mask = labels[:, 0] == class_idx
        corrupted_instances = instances[class_mask][labels[class_mask][:, 0] != labels[class_mask][:, 1]]
        corrupted_mean = np.mean(corrupted_instances, axis=0)

        loc_idx = 0
        for other_class_idx in classes:
            if other_class_idx == class_idx:
                continue

            corrupted_vector = corrupted_mean - class_means[class_idx]
            cls_to_cls_centroid = class_means[other_class_idx] - class_means[class_idx]

            tmp_bias = np.dot(corrupted_vector, cls_to_cls_centroid) / np.linalg.norm(cls_to_cls_centroid)
            corruption_biases[
                class_idx, loc_idx] = tmp_bias  # max(tmp_bias, np.dot(cls_to_cls_centroid, corrupted_vector))
            loc_idx += 1
        logging.info("Corruption biases: {}".format(corruption_biases))

    return corruption_biases


def update_accuracy(top1, inputs, outputs, targets, is_binary):
    if is_binary:
        prec1 = compute_binary_accuracy(outputs.detach().data, targets.detach().data)
        top1.update(prec1, inputs.size(0))
    else:
        prec1, _ = compute_accuracy(outputs.detach().data, targets.detach().data, topk=(1, 2))
        top1.update(prec1.item(), inputs.size(0))


def relative_mse(predictions, targets):
    return torch.mean(torch.square(predictions - targets) / (targets + 1e-7))


def relative_mae(predictions, targets):
    return torch.mean(torch.abs(predictions - targets) / (targets + 1e-7))
