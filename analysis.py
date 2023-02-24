import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from args import load_args_dict
from validate_NC import initialize_info_dict, eval_model, FCFeatures, FCOutputs
from viz.eval_simple_model_viz import evaluate_model_visually
from viz.metrics import get_class_means, get_class_variances, get_centroids_by_corruption
from viz.viz_utils import CLASS_COLORS, DIGIT_CLASS_MARKERS, visualize_class_cluster_dispersion_2d, get_pp_wd_str, \
    get_pp_bn_str, get_pp_loss_str, visualize_centroids_by_corruption


def dynamics_analysis(args, model, trainloader, testloader, num_classes, epoch_id, visualize=True):
    model.eval()

    # Track metrics
    info_dict = initialize_info_dict(num_eval_classes=min(num_classes, 5))

    fc_features = FCFeatures()
    pre_hook_handle = model.fc.register_forward_pre_hook(fc_features)

    fc_postsoftmax = FCOutputs()
    forward_hook_handle = model.fc.register_forward_hook(fc_postsoftmax)

    logfile = open('%s/analysis_log.txt' % (args.save_path), 'a')
    with torch.no_grad():
        eval_model(args, model, info_dict, fc_features, fc_postsoftmax, trainloader, testloader, epoch_id,
                   logfile, num_eval_classes=num_classes)

    pre_hook_handle.remove()
    del fc_features

    forward_hook_handle.remove()
    del fc_postsoftmax

    info_dir_path = os.path.join(args.save_path, "infos")
    os.makedirs(info_dir_path, exist_ok=True)
    with open(os.path.join(info_dir_path, 'info_{}.pkl'.format(epoch_id)), 'wb') as f:
        pickle.dump(info_dict, f)

    if visualize and num_classes <= 10 and args.model in ["SimpleMLP", "MLP"]:
        # Plot feature activations
        evaluate_model_visually(args, model, base_path=args.save_path, plot_suffix="_{}".format(epoch_id),
                                output_dir=os.path.join(args.save_path, "plots"), file_extension="pdf",
                                epoch_id=epoch_id, show_plots=False, store_data=True,
                                store_data_prefix="infos/activations")

    logfile.close()

    torch.cuda.empty_cache()
    model.train()


def plt_stddevs(train_activations, train_labels, test_activations, test_labels, classes, num_classes, args, base_path):
    # Idea is to see how the training evolves over the course of the training
    fig, ax = plt.subplots(figsize=(8, 6))

    train_class_0 = np.zeros(train_activations.shape[0])
    train_class_1 = np.zeros(train_activations.shape[0])
    train_class_avg = np.zeros(train_activations.shape[0])
    train_uncorr_0 = np.zeros(train_activations.shape[0])
    train_uncorr_1 = np.zeros(train_activations.shape[0])
    train_corr_0 = np.zeros(train_activations.shape[0])
    train_corr_1 = np.zeros(train_activations.shape[0])

    test_class_0 = np.zeros(train_activations.shape[0])
    test_class_1 = np.zeros(train_activations.shape[0])
    test_class_avg = np.zeros(train_activations.shape[0])

    # Read infos
    for epoch_idx in range(train_activations.shape[0]):
        # Train instances
        train_instances = train_activations[epoch_idx]
        train_targets = train_labels[epoch_idx]

        train_means = get_class_means(train_instances, train_targets, classes, num_features=num_classes)
        _, train_stddevs = get_class_variances(train_instances, train_targets, classes, train_means)

        train_class_0[epoch_idx] = train_stddevs[0]
        train_class_1[epoch_idx] = train_stddevs[1]
        train_class_avg[epoch_idx] = 0.5 * (train_stddevs[0] + train_stddevs[1])

        uncorrupted_mask = train_targets[:, 0] == train_targets[:, 1]
        corrupted_mask = np.logical_not(uncorrupted_mask)

        uncorrupted_centroids, tmp_corrupted_centroids = get_centroids_by_corruption(args, train_instances,
                                                                                     train_targets,
                                                                                     classes, num_features=num_classes)

        assert num_classes == 2, "Currently, only the two-class case is supported!"
        corrupted_centroids = np.zeros([2, 2])
        corrupted_centroids[0] = tmp_corrupted_centroids[1, 0]
        corrupted_centroids[1] = tmp_corrupted_centroids[0, 1]

        _, train_uncorr_stddevs = get_class_variances(train_instances[uncorrupted_mask],
                                                      train_targets[uncorrupted_mask], classes, uncorrupted_centroids)
        _, train_corr_stddevs = get_class_variances(train_instances[corrupted_mask], train_targets[corrupted_mask],
                                                    classes, corrupted_centroids)
        train_uncorr_0[epoch_idx] = train_uncorr_stddevs[0]
        train_uncorr_1[epoch_idx] = train_uncorr_stddevs[1]

        train_corr_0[epoch_idx] = train_corr_stddevs[0]
        train_corr_1[epoch_idx] = train_corr_stddevs[1]

        # Test instances
        test_instances = test_activations[epoch_idx]
        test_targets = test_labels[epoch_idx]

        test_means = get_class_means(test_instances, test_targets, classes, num_features=num_classes)
        _, test_stddevs = get_class_variances(test_instances, test_targets, classes, test_means)

        test_class_0[epoch_idx] = test_stddevs[0]
        test_class_1[epoch_idx] = test_stddevs[1]
        test_class_avg[epoch_idx] = 0.5 * (test_stddevs[0] + test_stddevs[1])

    # Plot
    x = np.linspace(0, train_activations.shape[0], train_activations.shape[0])
    ax.plot(x, train_class_0, label="Train class 0", linestyle="solid", color="tab:green")
    ax.plot(x, test_class_0, label="Test class 0", linestyle="dashed", color="tab:green")

    wd_str = get_pp_wd_str(args.weight_decay)
    bn_str = get_pp_bn_str(args.use_bn)
    loss_str = get_pp_loss_str(args.loss)
    ax.set_title(
        "{} ({}x{} | {} | LN: {} | {} | {})".format(loss_str, args.depth, args.width, args.act_fn,
                                                    args.label_noise, bn_str, wd_str))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Standard deviation")

    # ax.legend()
    ax.legend(loc="center left", fancybox=True, bbox_to_anchor=(1.0, 0.5), ncol=1, shadow=True)

    fig.savefig(os.path.join(base_path, "stddevs.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(base_path, "stddevs.png"), bbox_inches="tight")
    plt.subplots_adjust(right=0.75)
    plt.show(block=False)
    # plt.clf()


def add_activations_and_labels(activations, labels, path):
    data = np.load(path)
    tmp_activations = data["instances"]
    tmp_labels = data["labels"]

    if activations is None:
        activations = tmp_activations
        activations = np.expand_dims(activations, axis=0)
    else:
        activations = np.concatenate((activations, np.expand_dims(tmp_activations, 0)), 0)

    if labels is None:
        labels = tmp_labels
        labels = np.expand_dims(labels, axis=0)
    else:
        labels = np.concatenate((labels, np.expand_dims(tmp_labels, 0)), 0)

    return activations, labels


def construct_interactive_plot(base_path, plot_stddevs):
    # Create interactive plot of the activations over the course of the training

    # Read args to print meta information about the run
    args_path = os.path.join(base_path, "args.json")
    args = load_args_dict(args_path)
    epochs = args.epochs

    classes = [0, 1]
    num_classes = len(classes)

    # Slides: # epochs, feature 1 max value, feature 2 max value
    # Read in all activations
    base_path = os.path.join(base_path, "infos")
    train_activations, train_labels = None, None
    test_activations, test_labels = None, None
    for i in range(epochs):
        train_activations, train_labels = add_activations_and_labels(train_activations, train_labels,
                                                                     os.path.join(base_path,
                                                                                  "activations_train_{}.npz".format(i)))

        test_activations, test_labels = add_activations_and_labels(test_activations, test_labels,
                                                                   os.path.join(base_path,
                                                                                "activations_test_{}.npz".format(i)))

    if plot_stddevs:
        plt_stddevs(train_activations, train_labels, test_activations, test_labels, classes, num_classes, args,
                    base_path)

    # Meta data
    num_plotted_instances = 1000
    marker_size = 150
    font_size = 14
    alpha = 0.5

    init_epoch_id = 0
    init_feature1_max = 3.5
    init_feature2_max = 3.5

    fig, axes = plt.subplots(figsize=(16, 6), ncols=2)

    def format_ax(ax, split_name):
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

        wd_str = get_pp_wd_str(args.weight_decay)
        bn_str = get_pp_bn_str(args.use_bn)
        loss_str = get_pp_loss_str(args.loss)

        ax.set_title(
            "{} ({}x{} | {} | LN: {} | {} | {} | {})".format(loss_str, args.depth, args.width, args.act_fn,
                                                             args.label_noise, bn_str, wd_str, split_name))

    # Build plot
    plt.subplots_adjust(left=0.25, bottom=0.25)
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
    epoch_slider = Slider(
        ax=axfreq,
        label='Epoch',
        valmin=0,
        valmax=args.epochs - 1,
        valinit=init_epoch_id, valfmt="%i"
    )
    axfreq = plt.axes([0.25, 0.15, 0.65, 0.03])
    feature1_slider = Slider(
        ax=axfreq,
        label='Feature 1 Max',
        valmin=0.1,
        valmax=10,
        valinit=init_feature1_max,
    )
    axspeed = plt.axes([0.25, 0.05, 0.65, 0.03])
    speed_slider = Slider(
        ax=axspeed,
        label='Animation speed (s)',
        valmin=0.1,
        valmax=1.0,
        valinit=0.5,
    )

    axamp = plt.axes([0.1, 0.25, 0.0225, 0.63])
    feature2_slider = Slider(
        ax=axamp,
        label="Feature 2 Max",
        valmin=0.1,
        valmax=10,
        valinit=init_feature2_max,
        orientation="vertical"
    )

    def update_plot(val):
        train_ax = axes[0]
        test_ax = axes[1]

        plt.sca(train_ax)
        train_ax.clear()

        format_ax(train_ax, "Train")

        epoch_id, feature1_max, feature2_max = int(epoch_slider.val), feature1_slider.val, feature2_slider.val

        # Plot training instances
        # Scatter plot instances
        instances = np.squeeze(train_activations[epoch_id])
        labels = np.squeeze(train_labels[epoch_id])

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
            plt.scatter(
                instances[:, 0][corrupted_mask][labels[corrupted_mask][:, 1] == class_idx][:num_plotted_instances],
                instances[:, 1][corrupted_mask][labels[corrupted_mask][:, 1] == class_idx][:num_plotted_instances],
                marker=DIGIT_CLASS_MARKERS[class_idx], c="tab:red", label="Class {} (corrupted)".format(class_idx),
                alpha=alpha)

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
        visualize_class_cluster_dispersion_2d(classes, std_devs, class_means, alpha=0.3)

        if type(args.label_noise) in [np.ndarray, list] or args.label_noise > 0.0:
            uncorrupted_centroids, corrupted_centroids = get_centroids_by_corruption(args, instances, labels, classes,
                                                                                     num_features=num_classes)
            visualize_centroids_by_corruption(uncorrupted_centroids, corrupted_centroids, classes, marker_size)

        plt.xlim(min(np.min(instances[:, 0]), -0.01), feature1_max)
        plt.ylim(min(np.min(instances[:, 1]), -0.01), feature2_max)

        plt.legend(loc="center left", fancybox=True, bbox_to_anchor=(2.25, 0.5), ncol=1, shadow=True)

        # Plot test instances
        plt.sca(test_ax)
        test_ax.clear()
        format_ax(test_ax, "Test")

        test_instances = np.squeeze(test_activations[epoch_id])
        test_targets = np.squeeze(test_labels[epoch_id])

        for class_idx in classes:
            # Plot test instances
            plt.scatter(test_instances[:, 0][test_targets == class_idx],
                        test_instances[:, 1][test_targets == class_idx],
                        marker=DIGIT_CLASS_MARKERS[class_idx], c=CLASS_COLORS[class_idx],
                        label="Class {}".format(class_idx),
                        alpha=alpha)
        # Plot class means
        test_class_means = get_class_means(test_instances, test_targets, classes, num_features=num_classes)
        test_variances, test_std_devs = get_class_variances(test_instances, test_targets, classes, test_class_means)
        for class_idx in classes:
            plt.scatter(test_class_means[class_idx][0], test_class_means[class_idx][1], marker="*",
                        label="Class {} centroid".format(class_idx),
                        c=CLASS_COLORS[class_idx], edgecolors="black",
                        s=marker_size)
        visualize_class_cluster_dispersion_2d(classes, test_std_devs, test_class_means, alpha=0.3)

        plt.xlim(min(np.min(instances[:, 0]), -0.01), feature1_max)
        plt.ylim(min(np.min(instances[:, 1]), -0.01), feature2_max)

    epoch_slider.on_changed(update_plot)
    feature1_slider.on_changed(update_plot)
    feature2_slider.on_changed(update_plot)

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = plt.axes([0.8, 0.0, 0.15, 0.04])
    button = Button(resetax, 'Animation', hovercolor='0.975')
    import time
    import threading
    def animation(event):
        def update_plot():
            for i in range(args.epochs):
                epoch_slider.set_val(i)
                time.sleep(speed_slider.val)

        t1 = threading.Thread(target=update_plot)
        t1.start()

    button.on_clicked(animation)

    update_plot(None)
    plt.subplots_adjust(right=0.8)
    plt.show(block=True)


if __name__ == "__main__":
    import sys

    assert len(sys.argv) == 2, "the path must be specified!"

    path = str(sys.argv[1])
    plot_stddevs = True
    construct_interactive_plot(path, plot_stddevs)
