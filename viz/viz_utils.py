from typing import List
import os

import numpy as np
import matplotlib.pyplot as plt

from utils.utils import CROSS_ENTROPY_TAG

BASIC_COLORS: List[str] = ["green", "red", "blue", "orange", "purple", "yellow", "black", "pink", "brown", "gray"]
CLASS_COLORS: List[str] = ["tab:green", "tab:blue", "tab:brown", "tab:orange", "tab:pink", "tab:purple", "tab:olive",
                           "tab:gray", "tab:cyan", "tab:red"]
CLASS_MARKERS: List[str] = ["+", "x", "1", ".", "|", "_", "*", "4", "D", "^"]

DIGIT_CLASS_MARKERS: List[str] = ["${}$".format(i) for i in range(10)]

XTICKS = [0, 50, 100, 150, 200]

RUN_MARKERS: List[str] = ['v', 'o', 's', 'X', '.']
RUN_COLORS: List[str] = ['r', 'b', 'g', 'm', 'y']


def visualize_class_cluster_dispersion_2d(classes, std_devs, class_means, alpha=0.1):
    for class_idx in classes:
        loc_circle = plt.Circle((class_means[class_idx, 0], class_means[class_idx, 1]), std_devs[class_idx],
                                color=CLASS_COLORS[class_idx],
                                alpha=alpha)
        plt.gca().add_patch(loc_circle)


def visualize_clf_2d(model, instances, is_binary, num_classes, label="Classifier"):
    fc_x_lin_space = np.linspace(np.min(instances[:, 0]), np.max(instances[:, 0]), 100)
    if is_binary:
        coefficients = np.squeeze(model.fc.weight.cpu().detach().numpy())
        bias = model.fc.bias.cpu().detach().numpy()

        def clf(loc_x):
            return (bias[0] + coefficients[0] * loc_x) / -coefficients[1]

        plt.plot(fc_x_lin_space, clf(fc_x_lin_space), label=label, linestyle="--", c="black", linewidth=3)
    elif num_classes == 2:
        # Two class softmax
        coefficients = np.squeeze(model.fc.weight.cpu().detach().numpy())
        biases = model.fc.bias.cpu().detach().numpy()

        def clf(loc_x):
            return (biases[0] - biases[1] + (coefficients[0, 0] - coefficients[1, 0]) * loc_x) / \
                   (coefficients[1, 1] - coefficients[0, 1])

        plt.plot(fc_x_lin_space, clf(fc_x_lin_space), label=label, linestyle="--", c="black", linewidth=3)
    else:

        print("Currently, decision boundary plotting is not implemented for softmax with more than two classes.")


def visualize_centroids_by_corruption(uncorrupted_centroids, corrupted_centroids, classes, marker_size):
    # Separate for loops for more intuitive order in the legend
    for class_idx in classes:
        plt.scatter(uncorrupted_centroids[class_idx, 0], uncorrupted_centroids[class_idx, 1],
                    # marker="*", c=CLASS_COLORS[class_idx],
                    marker=r"$\mathbf{%s}$" % class_idx, c=CLASS_COLORS[class_idx], linewidth=1.,
                    edgecolors="black",
                    label="Uncorrupt. {} centroid".format(class_idx), s=marker_size)

    for inner_class_idx in classes:
        for class_idx in classes:
            if class_idx == inner_class_idx:
                continue
            plt.scatter(corrupted_centroids[class_idx, inner_class_idx, 0],
                        corrupted_centroids[class_idx, inner_class_idx, 1],
                        # marker="d", c=CLASS_COLORS[class_idx],
                        # marker=DIGIT_CLASS_MARKERS[inner_class_idx], c=BASIC_COLORS[1], linewidth=0.25,
                        marker=r"$\mathbf{%s}$" % inner_class_idx, c=BASIC_COLORS[1], linewidth=1.,
                        label="Corrupt. {} centroid".format(inner_class_idx),
                        s=marker_size, edgecolors="black")


def format_scatter_axes_2d(instances):
    plt.legend()
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.xlim(min(np.min(instances[:, 0]), -0.01), max(0.01, np.max(instances[:, 0])))
    plt.ylim(min(np.min(instances[:, 1]), -0.01), max(0.01, np.max(instances[:, 1])))


def get_pp_loss_str(loss):
    if loss == CROSS_ENTROPY_TAG:
        loss_str = "CE"
    else:
        loss_str = loss
    return loss_str


def get_pp_bn_str(use_bn):
    if use_bn:
        bn_str = "with BN"
    else:
        bn_str = "No BN"
    return bn_str


def get_pp_wd_str(weight_decay):
    if weight_decay == 0.0:
        wd_str = "No WD"
    else:
        wd_str = "WD: {}".format(str(weight_decay))
    return wd_str

def get_pp_fd_str(args):
    feature_decay = args.feature_decay_rate
    if not args.sep_decay:
        feature_decay = 0.0

    if feature_decay == 0.0:
        fd_str = "No FD"
    else:
        fd_str = "FD: {}".format(str(feature_decay))
    return fd_str


def save_plot_as_pdf_png(output_dir, file_name, fig=None):
    final_path_pdf = os.path.join(output_dir, file_name + ".pdf")
    final_path_png = os.path.join(output_dir, file_name + ".png")
    if fig is None:
        plt.savefig(final_path_pdf, bbox_inches='tight')
        plt.savefig(final_path_png, bbox_inches='tight')
    else:
        fig.savefig(final_path_pdf, bbox_inches='tight')
        fig.savefig(final_path_png, bbox_inches='tight')


def initialize_old_exp_plots():
    plt.rcParams.update({'font.size': 30})


def aggregate_dict_list_results(dict_list, key_to_aggregate):
    results = []
    for dict_ in dict_list:
        results.append(dict_[key_to_aggregate])
    return np.mean(results, axis=0)

def aggregate_dict_list_results_stddev(dict_list, key_to_aggregate):
    results = []
    for dict_ in dict_list:
        results.append(dict_[key_to_aggregate])
    return np.mean(results, axis=0), np.std(results, axis=0)


def plot_collapse(infos_args, output_dir, file_path_prefix, legend=None):
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    for i in range(len(infos_args)):
        info_arg = infos_args[i][0]
        if isinstance(info_arg, list):
            value = aggregate_dict_list_results(info_arg, "collapse_metric")
        else:
            value = info_arg["collapse_metric"]

        plt.plot(value, RUN_COLORS[i], marker=RUN_MARKERS[i], ms=16, markevery=25, linewidth=5,
                 alpha=0.7)

    if legend is not None:
        plt.legend(legend)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_1$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    save_plot_as_pdf_png(output_dir, file_path_prefix + "NC1", fig=fig)


def plot_collapse_post(infos_args, output_dir, file_path_prefix, legend=None):
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    for i in range(len(infos_args)):
        info_arg = infos_args[i][0]
        if isinstance(info_arg, list):
            value = aggregate_dict_list_results(info_arg, "collapse_metric_post")
        else:
            value = info_arg["collapse_metric_post"]

        plt.plot(value, RUN_COLORS[i], marker=RUN_MARKERS[i], ms=16, markevery=25,
                 linewidth=5,
                 alpha=0.7)

    if legend is not None:
        plt.legend(legend)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_1$ (Post)', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    save_plot_as_pdf_png(output_dir, file_path_prefix + "NC1_post", fig=fig)


def plot_ETF(infos_args, output_dir, file_path_prefix, legend=None):
    # NC2
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    for i in range(len(infos_args)):
        info_arg = infos_args[i][0]
        if isinstance(info_arg, list):
            value = aggregate_dict_list_results(info_arg, "ETF_metric")
        else:
            value = info_arg["ETF_metric"]

        plt.plot(value, RUN_COLORS[i], marker=RUN_MARKERS[i], ms=16, markevery=25, linewidth=5,
                 alpha=0.7)

    if legend is not None:
        plt.legend(legend)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_2$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    save_plot_as_pdf_png(output_dir, file_path_prefix + "NC2", fig=fig)


def plot_WH_relation(infos_args, output_dir, file_path_prefix, legend=None):
    # NC3
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    for i in range(len(infos_args)):
        info_arg = infos_args[i][0]
        if isinstance(info_arg, list):
            value = aggregate_dict_list_results(info_arg, "WH_relation_metric")
        else:
            value = info_arg["WH_relation_metric"]

        plt.plot(value, RUN_COLORS[i], marker=RUN_MARKERS[i], ms=16, markevery=25,
                 linewidth=5,
                 alpha=0.7)

    if legend is not None:
        plt.legend(legend)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_3$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    save_plot_as_pdf_png(output_dir, file_path_prefix + "NC3", fig=fig)


def plot_residual(infos_args, output_dir, file_path_prefix, legend=None):
    # NC4
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    for i in range(len(infos_args)):
        info_arg = infos_args[i][0]
        if isinstance(info_arg, list):
            value = aggregate_dict_list_results(info_arg, "Wh_b_relation_metric")
        else:
            value = info_arg["Wh_b_relation_metric"]

        plt.plot(value, RUN_COLORS[i], marker=RUN_MARKERS[i], ms=16, markevery=25,
                 linewidth=5,
                 alpha=0.7)

    if legend is not None:
        plt.legend(legend)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_4$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    save_plot_as_pdf_png(output_dir, file_path_prefix + "NC4", fig=fig)


def plot_train_acc(infos_args, output_dir, file_path_prefix, legend=None):
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    for i in range(len(infos_args)):
        info_arg = infos_args[i][0]
        if isinstance(info_arg, list):
            value = aggregate_dict_list_results(info_arg, "train_acc1")
        else:
            value = info_arg["train_acc1"]

        plt.plot(value, RUN_COLORS[i], marker=RUN_MARKERS[i], ms=16, markevery=25, linewidth=5,
                 alpha=0.7)

    if legend is not None:
        plt.legend(legend)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Training accuracy', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    save_plot_as_pdf_png(output_dir, file_path_prefix + "train-acc", fig=fig)


def plot_test_acc(infos_args, output_dir, file_path_prefix, legend=None):
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    for i in range(len(infos_args)):
        info_arg = infos_args[i][0]
        if isinstance(info_arg, list):
            value = aggregate_dict_list_results(info_arg, "test_acc1")
        else:
            value = info_arg["test_acc1"]

        plt.plot(value, RUN_COLORS[i], marker=RUN_MARKERS[i], ms=16, markevery=25, linewidth=5,
                 alpha=0.7)

    if legend is not None:
        plt.legend(legend)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Testing accuracy', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    save_plot_as_pdf_png(output_dir, file_path_prefix + "test-acc", fig=fig)


def plot_ECE_train(infos_args, output_dir, file_path_prefix, legend=None):
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    for i in range(len(infos_args)):
        info_arg = infos_args[i][0]
        if isinstance(info_arg, list):
            value = aggregate_dict_list_results(info_arg, "ece_metric_train")
        else:
            value = info_arg["ece_metric_train"]

        plt.plot(value, RUN_COLORS[i], marker=RUN_MARKERS[i], ms=16, markevery=25,
                 linewidth=5,
                 alpha=0.7)

    plt.yscale("logit")
    plt.yticks(fontsize=30)

    if legend is not None:
        plt.legend(legend)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Training ECE', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    save_plot_as_pdf_png(output_dir, file_path_prefix + "train-ece", fig=fig)


def plot_ECE_test(infos_args, output_dir, file_path_prefix, legend=None):
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    for i in range(len(infos_args)):
        info_arg = infos_args[i][0]
        if isinstance(info_arg, list):
            value = aggregate_dict_list_results(info_arg, "ece_metric_test")
        else:
            value = info_arg["ece_metric_test"]

        plt.plot(value, RUN_COLORS[i], marker=RUN_MARKERS[i], ms=16, markevery=25, linewidth=5,
                 alpha=0.7)

    plt.yscale("logit")
    plt.yticks(fontsize=30)

    if legend is not None:
        plt.legend(legend)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Testing ECE', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    save_plot_as_pdf_png(output_dir, file_path_prefix + "test-ece", fig=fig)


def plot_sigma_W(infos_args, output_dir, file_path_prefix, legend=None):
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    for i in range(len(infos_args)):
        info_arg = infos_args[i][0]
        if isinstance(info_arg, list):
            sigma_W = np.zeros([len(info_arg), len(info_arg[0]['Sigma_W'])])
            for k in range(len(info_arg)):
                for j in range(sigma_W.shape[1]):
                    sigma_W[k, j] = np.trace(info_arg[k]['Sigma_W'][j])

            value = np.mean(sigma_W, axis=0)

        else:
            value = []
            for elem in info_arg['Sigma_W']:
                value.append(np.trace(elem))

        plt.plot(value, RUN_COLORS[i], marker=RUN_MARKERS[i], ms=16, markevery=25, linewidth=5,
                 alpha=0.7)

    if legend is not None:
        plt.legend(legend)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('$tr(\\Sigma_W)$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    save_plot_as_pdf_png(output_dir, file_path_prefix + "sigma_W", fig=fig)


def plot_sigma_B(infos_args, output_dir, file_path_prefix, legend=None):
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    for i in range(len(infos_args)):
        info_arg = infos_args[i][0]
        if isinstance(info_arg, list):
            sigma_W = np.zeros([len(info_arg), len(info_arg[0]['Sigma_B'])])
            for k in range(len(info_arg)):
                for j in range(sigma_W.shape[1]):
                    sigma_W[k, j] = np.trace(info_arg[k]['Sigma_B'][j])

            value = np.mean(sigma_W, axis=0)

        else:
            value = []
            for elem in info_arg['Sigma_B']:
                value.append(np.trace(elem))

        plt.plot(value, RUN_COLORS[i], marker=RUN_MARKERS[i], ms=16, markevery=25, linewidth=5,
                 alpha=0.7)

    if legend is not None:
        plt.legend(legend)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('$tr(\\Sigma_B)$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    save_plot_as_pdf_png(output_dir, file_path_prefix + "sigma_B", fig=fig)
