import sys
import pickle
import scipy.linalg as scilin

import models
from utils.utils import *
from args import parse_train_args
from datasets import make_reproducible_dataset, AugmentedDataset

import torch.nn.functional as F


class FCFeatures:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in):
        self.outputs.append(module_in)

    def clear(self):
        self.outputs = []


class FCOutputs:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, input, output):
        # TODO: Apply softmax here?
        softmax_output = output.softmax(dim=-1)

        self.outputs.append(softmax_output)

    def clear(self):
        self.outputs = []


def compute_info(args, model, fc_features, fc_features_post, dataloader, is_train=True, num_eval_classes=5):
    mu_G = 0
    mu_G_post = 0
    mu_c_dict = dict()
    mu_c_dict_post = dict()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Probabilistic predictions


    if is_train:
        samples = np.zeros(args.classes, dtype=np.int32)
    else:
        num_test_classes = args.classes
        samples = np.zeros(num_test_classes, dtype=np.int32)

    multi_input = isinstance(dataloader.dataset, AugmentedDataset)
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if multi_input:
            inputs = inputs[0]
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        with torch.no_grad():
            outputs = model(inputs)

        features = fc_features.outputs[0][0]
        fc_features.clear()

        post_sm_activations = fc_features_post.outputs[0]
        fc_features_post.clear()

        mu_G += torch.sum(features, dim=0)
        mu_G_post += torch.sum(post_sm_activations, dim=0)

        for b in range(len(targets)):
            y = targets[b].item()
            if y not in mu_c_dict:
                mu_c_dict[y] = features[b, :]
            else:
                mu_c_dict[y] += features[b, :]

            if y not in mu_c_dict_post:
                mu_c_dict_post[y] = post_sm_activations[b, :]
            else:
                mu_c_dict_post[y] += post_sm_activations[b, :]

            samples[y] += 1

        prec1, prec5 = compute_accuracy(outputs.data, targets.data, topk=(1, num_eval_classes))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

    # Normalize elements
    if type(args.label_noise) in [np.ndarray, list]:
        train_samples, test_samples = samples, samples
    # elif not args.fourclass_problem:
    #     train_samples, test_samples = get_train_test_samples_per_dataset_2_or_4c(args)
    else:
        train_samples, test_samples = samples, samples

    if is_train:
        mu_G /= sum(train_samples)
        mu_G_post /= sum(train_samples)
        for i in range(len(train_samples)):
            mu_c_dict[i] /= train_samples[i]
            mu_c_dict_post[i] /= train_samples[i]
    else:
        mu_G /= sum(test_samples)
        mu_G_post /= sum(test_samples)
        for i in range(len(test_samples)):
            mu_c_dict[i] /= test_samples[i]
            mu_c_dict_post[i] /= test_samples[i]

    return mu_G, mu_c_dict, mu_G_post, mu_c_dict_post, top1.avg, top5.avg


def compute_Sigma_W(args, model, fc_features, mu_c_dict, fc_postsoftmax, mu_c_dict_post, dataloader, is_train=True):
    Sigma_W = 0
    Sigma_W_post = 0

    if is_train:
        samples = np.zeros(args.classes, dtype=np.int32)
    else:
        num_test_classes = args.classes
        samples = np.zeros(num_test_classes, dtype=np.int32)

    multi_input = isinstance(dataloader.dataset, AugmentedDataset)
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if multi_input:
            inputs = inputs[0]
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        with torch.no_grad():
            outputs = model(inputs)

        features = fc_features.outputs[0][0]
        fc_features.clear()

        post_sm_activations = fc_postsoftmax.outputs[0]
        fc_postsoftmax.clear()

        for b in range(len(targets)):
            y = targets[b].item()

            Sigma_W += (features[b, :] - mu_c_dict[y]).unsqueeze(1) @ (features[b, :] - mu_c_dict[y]).unsqueeze(0)
            Sigma_W_post += (post_sm_activations[b, :] - mu_c_dict_post[y]).unsqueeze(1) @ (
                    post_sm_activations[b, :] - mu_c_dict_post[y]).unsqueeze(0)

            samples[y] += 1

    if type(args.label_noise) in [np.ndarray, list]:
        train_samples, test_samples = samples, samples
    # elif not args.fourclass_problem:
    #     train_samples, test_samples = get_train_test_samples_per_dataset_2_or_4c(args)
    else:
        train_samples, test_samples = samples, samples

    if is_train:
        Sigma_W /= sum(train_samples)
        Sigma_W_post /= sum(train_samples)
    else:
        Sigma_W /= sum(test_samples)
        Sigma_W_post /= sum(test_samples)

    return Sigma_W.cpu().numpy(), Sigma_W_post.cpu().numpy()


def compute_Sigma_B(mu_c_dict, mu_G):
    Sigma_B = 0
    K = len(mu_c_dict)
    for i in range(K):
        Sigma_B += (mu_c_dict[i] - mu_G).unsqueeze(1) @ (mu_c_dict[i] - mu_G).unsqueeze(0)

    Sigma_B /= K

    return Sigma_B.cpu().numpy()


def compute_ETF(W):
    K = W.shape[0]
    WWT = torch.mm(W, W.T)
    WWT /= torch.norm(WWT, p='fro')

    sub = (torch.eye(K) - 1 / K * torch.ones((K, K))).cuda() / pow(K - 1, 0.5)
    ETF_metric = torch.norm(WWT - sub, p='fro')
    return ETF_metric.detach().cpu().numpy().item()


def compute_W_H_relation(W, mu_c_dict, mu_G):
    K = len(mu_c_dict)
    H = torch.empty(mu_c_dict[0].shape[0], K)
    for i in range(K):
        H[:, i] = mu_c_dict[i] - mu_G

    WH = torch.mm(W, H.cuda())
    WH /= torch.norm(WH, p='fro')
    sub = 1 / pow(K - 1, 0.5) * (torch.eye(K) - 1 / K * torch.ones((K, K))).cuda()

    res = torch.norm(WH - sub, p='fro')
    return res.detach().cpu().numpy().item(), H


def compute_Wh_b_relation(W, mu_G, b):
    Wh = torch.mv(W, mu_G.cuda())
    res_b = torch.norm(Wh - b, p='fro')
    return res_b.detach().cpu().numpy().item()


def compute_ECE(args, model, dataloader, fc_features, n_bins=15):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # Compute expected calibration error
    logits_list = []
    labels_list = []

    multi_input = isinstance(dataloader.dataset, AugmentedDataset)
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Targets store potentially corrupted and original label
        if multi_input:
            inputs = inputs[0]
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        with torch.no_grad():
            outputs = model(inputs)

        logits_list.append(F.softmax(outputs.data, dim=-1))
        labels_list.append(targets.data)

    # Create tensors
    logits_list = torch.cat(logits_list).to(args.device)
    labels_list = torch.cat(labels_list).to(args.device)

    confidences, predictions = torch.max(logits_list, 1)
    accuracies = predictions.eq(labels_list)

    ece = torch.zeros(1, device=args.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    fc_features.clear()

    return ece.detach().cpu().numpy()


def eval_model(args, model, info_dict, fc_features, fc_postsoftmax, trainloader, testloader, epoch, logfile,
               num_eval_classes=5, model_path=None):
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    model.eval()

    for n, p in model.named_parameters():
        if 'fc.weight' in n:
            W = p
        if 'fc.bias' in n:
            b = p

    mu_G_train, mu_c_dict_train, mu_G_post_train, mu_c_dict_post_train, train_acc1, train_acc5 = compute_info(args,
                                                                                                              model,
                                                                                                              fc_features,
                                                                                                              fc_postsoftmax,
                                                                                                              trainloader,
                                                                                                              is_train=True,
                                                                                                              num_eval_classes=num_eval_classes)
    mu_G_test, mu_c_dict_test, mu_G_post_test, mu_c_dict_post_test, test_acc1, test_acc5 = compute_info(args, model,
                                                                                                        fc_features,
                                                                                                        fc_postsoftmax,
                                                                                                        testloader,
                                                                                                        is_train=False,
                                                                                                        num_eval_classes=num_eval_classes)

    Sigma_W, Sigma_W_post = compute_Sigma_W(args, model, fc_features, mu_c_dict_train, fc_postsoftmax,
                                            mu_c_dict_post_train, trainloader, is_train=True)
    info_dict['Sigma_W'].append(Sigma_W)
    info_dict['Sigma_W_post'].append(Sigma_W_post)

    # Sigma_W_test_norm = compute_Sigma_W(args, model, fc_features, mu_c_dict_train, testloader, isTrain=False)
    Sigma_B = compute_Sigma_B(mu_c_dict_train, mu_G_train)
    Sigma_B_post = compute_Sigma_B(mu_c_dict_post_train, mu_G_post_train)

    info_dict['Sigma_B'].append(Sigma_B)
    info_dict['Sigma_B_post'].append(Sigma_B_post)

    collapse_metric = np.trace(Sigma_W @ scilin.pinv(Sigma_B)) / len(mu_c_dict_train)
    collapse_metric_post = np.trace(Sigma_W_post @ scilin.pinv(Sigma_B_post)) / len(mu_c_dict_post_train)

    ETF_metric = compute_ETF(W)
    WH_relation_metric, H = compute_W_H_relation(W, mu_c_dict_train, mu_G_train)
    if args.bias:
        Wh_b_relation_metric = compute_Wh_b_relation(W, mu_G_train, b)
    else:
        Wh_b_relation_metric = compute_Wh_b_relation(W, mu_G_train, torch.zeros((W.shape[0],)))

    ece_metric_train = compute_ECE(args, model, trainloader, fc_features)
    ece_metric_test = compute_ECE(args, model, testloader, fc_features)

    info_dict['collapse_metric'].append(collapse_metric)
    info_dict['collapse_metric_post'].append(collapse_metric_post)
    info_dict['ETF_metric'].append(ETF_metric)
    info_dict['WH_relation_metric'].append(WH_relation_metric)
    info_dict['Wh_b_relation_metric'].append(Wh_b_relation_metric)

    info_dict['ece_metric_train'].append(ece_metric_train)
    info_dict['ece_metric_test'].append(ece_metric_test)

    info_dict['W'].append((W.detach().cpu().numpy()))
    if args.bias:
        info_dict['b'].append(b.detach().cpu().numpy())
    info_dict['H'].append(H.detach().cpu().numpy())

    info_dict['mu_G_train'].append(mu_G_train.detach().cpu().numpy())
    info_dict['mu_G_post_train'].append(mu_G_post_train.detach().cpu().numpy())

    info_dict['train_acc1'].append(train_acc1)
    info_dict['train_acc{}'.format(min(num_eval_classes, 5))].append(train_acc5)
    info_dict['test_acc1'].append(test_acc1)
    info_dict['test_acc{}'.format(min(num_eval_classes, 5))].append(test_acc5)

    print_and_save(
        '[epoch: %d] | train top1: %.4f | train top5: %.4f | test top1: %.4f | test top5: %.4f | train ECE: %.4f | test ECE: %.4f ' %
        (epoch, train_acc1, train_acc5, test_acc1, test_acc5, ece_metric_train, ece_metric_test), logfile)


def initialize_info_dict(num_eval_classes=5):
    return {
        'collapse_metric': [],
        'collapse_metric_post': [],
        'ETF_metric': [],
        'WH_relation_metric': [],
        'Wh_b_relation_metric': [],
        'W': [],
        'b': [],
        'H': [],
        'mu_G_train': [],
        'mu_G_post_train': [],
        # 'mu_G_test': [],
        'train_acc1': [],
        'train_acc{}'.format(num_eval_classes): [],
        'test_acc1': [],
        'test_acc{}'.format(num_eval_classes): [],
        'ece_metric_train': [],
        'ece_metric_test': [],

        # Additional metrics
        'Sigma_W': [],
        'Sigma_W_post': [],
        'Sigma_B': [],
        'Sigma_B_post': []
    }


def main():
    args = parse_train_args()

    args.load_path = args.save_path

    if args.load_path is None:
        sys.exit('Need to input the path to a pre-trained model!')

    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    args.device = device

    trainloader, _, testloader, num_classes = make_reproducible_dataset(args, args.load_path,
                                                                        label_noise=args.label_noise, eval=True)
    args.classes = num_classes

    if args.model in ["MLP", "SimpleMLP"]:
        model = models.__dict__[args.model](hidden=args.width, depth=args.depth, fc_bias=args.bias,
                                            num_classes=num_classes).to(device)
    else:
        model = models.__dict__[args.model](num_classes=num_classes, fc_bias=args.bias, ETF_fc=args.ETF_fc,
                                            fixdim=args.fixdim, SOTA=args.SOTA).to(device)

    fc_features = FCFeatures()
    model.fc.register_forward_pre_hook(fc_features)

    fc_postsoftmax = FCOutputs()
    model.fc.register_forward_hook(fc_postsoftmax)

    if os.path.exists(os.path.join(args.load_path, 'info.pkl')):
        print("Info already exists. Exiting...")
        sys.exit(0)

    info_dict = initialize_info_dict()

    logfile = open('%s/test_log.txt' % (args.load_path), 'w')
    for i in range(args.epochs):
        model_path = os.path.join(args.load_path, 'epoch_' + str(i + 1).zfill(3) + '.pth')
        eval_model(args, model, info_dict, fc_features, fc_postsoftmax, trainloader, testloader, i + 1,
                   logfile, model_path=model_path)

    with open(os.path.join(args.load_path, 'info.pkl'), 'wb') as f:
        pickle.dump(info_dict, f)


if __name__ == "__main__":
    main()
