from datasets import construct_preloaded_dataset, TRAIN_TARGETS_FN, TRAIN_ORIG_TARGETS_FN, TRAIN_DATA_FN, \
    SELECTED_LABELS_FN
from utils.utils import *
import matplotlib.pyplot as plt
import pickle
import logging

from validate_NC import FCFeatures
from viz.metrics import get_class_means, get_class_variances, CLUSTER_VAR_TAG, CLUSTER_STDDEV_TAG, CLUSTER_MEAN_TAG, \
    get_sigma_W, get_centroids_by_corruption, CLUSTER_UNCORRUPTED_STDDEV_TAG, get_uncorrupted_stddevs, \
    CLUSTER_UNCORRUPTED_MEAN_TAG, CLUSTER_CORRUPTED_MEAN_TAG, get_dot_product_bias, prepare_features_and_labels, \
    CORRUPTION_BIASES_TAG, update_accuracy
from viz.viz_utils import visualize_class_cluster_dispersion_2d, visualize_clf_2d, visualize_centroids_by_corruption, \
    format_scatter_axes_2d, CLASS_COLORS, DIGIT_CLASS_MARKERS, get_pp_loss_str, get_pp_bn_str, \
    get_pp_wd_str


def evaluate_model_visually(args, model, base_path, plot_suffix="", output_dir=None, file_extension="pdf",
                            epoch_id=None, store_data=False, store_data_prefix=None, show_plots=True):
    classes = [i for i in range(args.classes)]
    use_single_output = False
    is_binary = use_single_output

    assert base_path is not None
    train_targets = torch.load(os.path.join(base_path, TRAIN_TARGETS_FN))
    train_original_targets = torch.load(os.path.join(base_path, TRAIN_ORIG_TARGETS_FN))
    train_merged_targets = torch.stack((train_targets, train_original_targets), -1)
    train_data = torch.load(os.path.join(base_path, TRAIN_DATA_FN))
    selected_labels = torch.load(os.path.join(base_path, SELECTED_LABELS_FN))
    logging.debug("Selected labels: {}".format(selected_labels))

    trainloader, _, testloader, num_classes = construct_preloaded_dataset(args, train_data, train_merged_targets,
                                                                          selected_labels=[i for i in
                                                                                           range(args.classes)])

    fc_features = FCFeatures()
    pre_hook_handle = model.fc.register_forward_pre_hook(fc_features)

    # Test
    instances, labels = prepare_features_and_labels(model, trainloader, args, fc_features)

    # Prepare output
    instances = torch.cat(instances).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()

    if store_data:
        # Store data for plotting them later on
        assert store_data_prefix is not None
        assert epoch_id is not None

        np.savez(os.path.join(base_path, "{}_train_{}.npz".format(store_data_prefix, epoch_id)), instances=instances,
                 labels=labels)

    # Meta data
    num_plotted_instances = 250
    marker_size = 150
    font_size = 14
    alpha = 0.5

    # Scatter plot
    plt.rcParams.update({'font.size': font_size})
    _ = plt.figure(figsize=(8, 6))

    # For each class, use different marker and color
    uncorrupted_mask = labels[:, 0] == labels[:, 1]
    corrupted_mask = np.logical_not(uncorrupted_mask)

    for class_idx in classes:
        # Uncorrupted instances
        plt.scatter(
            instances[:, 0][uncorrupted_mask][labels[uncorrupted_mask][:, 0] == class_idx][:num_plotted_instances],
            instances[:, 1][uncorrupted_mask][labels[uncorrupted_mask][:, 0] == class_idx][:num_plotted_instances],
            marker=DIGIT_CLASS_MARKERS[class_idx], c=CLASS_COLORS[class_idx], label="Class {}".format(class_idx),
            alpha=alpha)

        # Corrupted instances
        plt.scatter(instances[:, 0][corrupted_mask][labels[corrupted_mask][:, 1] == class_idx][:num_plotted_instances],
                    instances[:, 1][corrupted_mask][labels[corrupted_mask][:, 1] == class_idx][:num_plotted_instances],
                    marker=DIGIT_CLASS_MARKERS[class_idx], c="tab:red", label="Class {} (corrupted)".format(class_idx),
                    alpha=alpha)

    # Plot classifier
    visualize_clf_2d(model, instances, is_binary, num_classes)

    wd_str = get_pp_wd_str(args.weight_decay)
    bn_str = get_pp_bn_str(args.use_bn)
    loss_str = get_pp_loss_str(args.loss)
    if epoch_id is None:
        epoch_str = args.epochs
    else:
        epoch_str = str(epoch_id)

    plt.title(
        "{} ({}x{} | {} | LN: {} | {} | {} | Train | E: {})".format(loss_str, args.depth, args.width, args.act_fn,
                                                                    args.label_noise, bn_str, wd_str, epoch_str))

    # Compute cluster variances
    class_means = get_class_means(instances, labels, classes, num_features=num_classes)
    variances, std_devs = get_class_variances(instances, labels, classes, class_means)

    # Plot class means
    for class_idx in classes:
        plt.scatter(class_means[class_idx][0], class_means[class_idx][1], marker="*",
                    label="Class {} centroid".format(class_idx),
                    c=CLASS_COLORS[class_idx], edgecolors="black",
                    s=marker_size)

    # Visualize the dispersion of the clusters
    visualize_class_cluster_dispersion_2d(classes, std_devs, class_means)

    # Compute Sigma_W
    Sigma_W = get_sigma_W(instances, labels, classes, class_means)

    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    logging.info("Variances: {}".format(variances))
    logging.info("Sigma_W: {}".format(Sigma_W))

    # Store all metrics
    metrics_dict = {CLUSTER_VAR_TAG: variances, CLUSTER_STDDEV_TAG: std_devs, CLUSTER_MEAN_TAG: class_means}
    if type(args.label_noise) in [np.ndarray, list] or args.label_noise > 0.0:
        uncorrupted_centroids, corrupted_centroids = get_centroids_by_corruption(args, instances, labels, classes,
                                                                                 num_features=num_classes)
        visualize_centroids_by_corruption(uncorrupted_centroids, corrupted_centroids, classes, marker_size)
        uncorrupted_stddevs = get_uncorrupted_stddevs(args, instances, labels, uncorrupted_centroids, classes)

        metrics_dict[CORRUPTION_BIASES_TAG] = get_dot_product_bias(instances, labels, classes, class_means)
        metrics_dict[CLUSTER_UNCORRUPTED_MEAN_TAG] = uncorrupted_centroids
        metrics_dict[CLUSTER_CORRUPTED_MEAN_TAG] = corrupted_centroids
        metrics_dict[CLUSTER_UNCORRUPTED_STDDEV_TAG] = uncorrupted_stddevs

    with open(os.path.join(base_path, 'metrics.pkl'), 'wb') as f:
        pickle.dump(metrics_dict, f)

    # We also have to measure the bias for class that is distributed in the coordinate origin
    format_scatter_axes_2d(instances)

    if output_dir is None:
        output_dir = base_path
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(os.path.join(output_dir, "training_cluster{}.{}".format(plot_suffix, file_extension)),
                bbox_inches="tight", transparent=True)
    if show_plots:
        plt.show(block=False)

    # Evaluate test performance
    top1 = AverageMeter()
    model.eval()

    test_instances = []
    test_labels = []

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        fc_features.clear()
        with torch.no_grad():
            outputs = model(inputs)

        features = fc_features.outputs[0][0]
        test_instances.append(features)
        test_labels.append(targets)

        update_accuracy(top1, inputs, outputs, targets, is_binary=is_binary)

    logging.info("Test performance: {}".format(top1.avg))

    test_instances = torch.cat(test_instances).cpu().numpy()
    test_labels = torch.cat(test_labels).cpu().numpy()

    if store_data:
        # Store data for plotting them later on
        assert store_data_prefix is not None
        assert epoch_id is not None

        np.savez(os.path.join(base_path, "{}_test_{}.npz".format(store_data_prefix, epoch_id)),
                 instances=test_instances, labels=test_labels)

    _ = plt.figure(figsize=(8, 6))

    for class_idx in classes:
        # Plot test instances
        plt.scatter(test_instances[:, 0][test_labels == class_idx], test_instances[:, 1][test_labels == class_idx],
                    marker=DIGIT_CLASS_MARKERS[class_idx], c=CLASS_COLORS[class_idx],
                    label="Class {}".format(class_idx),
                    alpha=alpha)

    visualize_clf_2d(model, test_instances, is_binary, num_classes)

    # Plot class means
    test_class_means = get_class_means(test_instances, test_labels, classes, num_features=num_classes)
    test_variances, test_std_devs = get_class_variances(test_instances, test_labels, classes, test_class_means)
    for class_idx in classes:
        plt.scatter(test_class_means[class_idx][0], test_class_means[class_idx][1], marker="*",
                    label="Class {} centroid".format(class_idx),
                    c=CLASS_COLORS[class_idx], edgecolors="black",
                    s=marker_size)
    visualize_class_cluster_dispersion_2d(classes, test_std_devs, test_class_means, alpha=0.3)

    plt.title(
        "{} ({}x{} | {} | LN: {} | {} | {} | Test | E: {})".format(loss_str, args.depth, args.width, args.act_fn,
                                                                   args.label_noise, bn_str, wd_str, epoch_str))

    format_scatter_axes_2d(instances)
    plt.savefig(os.path.join(output_dir, "test_cluster{}.{}".format(plot_suffix, file_extension)), bbox_inches="tight",
                transparent=True)
    if show_plots:
        plt.show(block=False)

    pre_hook_handle.remove()
    del fc_features
