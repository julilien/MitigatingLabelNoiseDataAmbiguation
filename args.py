import shutil
import datetime
import argparse
import json
import logging

from config_utils import get_config, set_log_level, get_base_path
from dataset_utils import *
from utils.io_utils import NumpyEncoder
from utils.simple_model_exps_utils import generate_uid


def add_default_parameter(parser):
    # Model Selection
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--no-bias', dest='bias', action='store_false')
    parser.add_argument('--ETF_fc', dest='ETF_fc', action='store_true')
    parser.add_argument('--fixdim', dest='fixdim', type=int, default=0)
    parser.add_argument('--SOTA', dest='SOTA', action='store_true')

    # MLP settings (only when using mlp and res_adapt(in which case only width has effect))
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--depth', type=int, default=6)

    # Hardware Setting
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--multi_gpu_ids', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use_cudnn', type=bool, default=True)
    parser.add_argument('--pin_memory', action="store_true")
    parser.add_argument('--num_workers', type=int, default=4)

    # Directory Setting
    parser.add_argument('--dataset', type=str,
                        choices=[MNIST_STR, CIFAR10_STR, FASHION_MNIST_STR, CIFAR100_STR, SVHN_STR,
                                 IMAGENET_STR, STL10_STR, CIFAR10N_CLEAN_STR, CIFAR10N_WORSE_STR, CIFAR10N_AGGRE_STR,
                                 CIFAR10N_RANDOM1_STR, CIFAR10N_RANDOM2_STR, CIFAR10N_RANDOM3_STR, CIFAR100N_CLEAN_STR,
                                 CIFAR100N_NOISY_STR, CLOTHING_1M_STR, WEBVISION_STR], default=MNIST_STR)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--uid', type=str, default=None)
    parser.add_argument('--force', action='store_true', help='force to override the given uid')

    # Learning Options
    parser.add_argument('--epochs', type=int, default=200, help='Max Epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--loss', type=str, default='CrossEntropy', help='loss function configuration')
    parser.add_argument('--sample_fraction', type=float, default=None, help='Fraction for sub-sampling.')

    # Optimization specifications
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--patience', type=int, default=40, help='learning rate decay per N epochs')
    parser.add_argument('--decay_type', type=str, default='step', choices=['step','cosine'], help='learning rate decay type')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor for step decay')
    parser.add_argument('--eta_min', type=float, default=0.0002, help='minimum learning rate for cosine decay')
    parser.add_argument('--optimizer', default='SGD', help='optimizer to use')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    # The following two should be specified when testing adding wd on Features
    parser.add_argument('--sep_decay', action='store_true',
                        help='whether to separate weight decay to last feature and last weights')
    parser.add_argument('--feature_decay_rate', type=float, default=1e-4, help='weight decay for last layer feature')
    parser.add_argument('--history_size', type=int, default=10, help='history size for LBFGS')
    parser.add_argument('--ghost_batch', type=int, dest='ghost_batch', default=128,
                        help='ghost size for LBFGS variants')

    parser.add_argument('--ls_alpha', type=float, default=0.1, help='Alpha parameter for label smoothing.')
    parser.add_argument('--lr_alpha', type=float, default=0.1, help='Alpha parameter for label relaxation.')

    # Second LR variant
    parser.add_argument('--lrvar2_warmup', type=int, default=0, help="Number of warmup periods.")
    parser.add_argument('--lrvar2_beta', type=float, default=0.2,
                        help="Beta parameter de-selecting plausibility (should be lower than lr_alpha).")
    parser.add_argument('--adaptive_lrvar2', action="store_true", help="Indicator whether to use adaptive beta.")
    parser.add_argument('--adaptive_lrvar2_start_beta', type=float, default=0.9)
    parser.add_argument('--adaptive_lrvar2_end_beta', type=float, default=0.9)
    parser.add_argument('--adaptive_lrvar2_type', type=str, default="linear", choices=["linear", "cosine"])


    parser.add_argument('--val_split_prop', type=float, default=0.0,
                        help='Split proportion used for validation set split.')
    parser.add_argument('--label_noise', type=float, default=0.0, help='Label noise used within the training data.')
    parser.add_argument('--use_asymmetric_noise', action="store_true", help="Indicator whether asymmetric noise shall "
                                                                            "be used (works only with datasets cifar10 "
                                                                            "and cifar100!).")
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--act_fn', default="relu", help="activation function of the penultimate layer")
    parser.add_argument('--use_bn', action='store_true', help="Use batch normalization")
    parser.add_argument('--use_layer_norm', action='store_true', help="Use batch normalization")
    parser.add_argument('--delete_model', action='store_true',
                        help="Indicator whether model should be deleted after validation.")

    parser.add_argument('--force_retrain', action='store_true',
                        help='Indicator for retraining an already available model.')

    parser.add_argument('--num_ho_runs', type=int, default=10)
    parser.add_argument('--ho', default='random_search')

    parser.add_argument('--use_largescale_uid', action='store_true', help='Shall we use the simple model UID scheme?')

    # MC noise sampling
    parser.add_argument('--sample_random_noise', action='store_true', help='We sample random label noise for each '
                                                                           'class individually. The instances are then '
                                                                           'subsampled to be roughly similarly '
                                                                           'distributed.')
    parser.add_argument('--max_sample_noise', type=float, default=0.5)

    # Learning dynamics parameters
    parser.add_argument('--trace_dynamics', action='store_true', help='Trace the learning behavior during the '
                                                                      'training.')
    parser.add_argument('--trace_freq', type=int, default=10, help="Modulo of the epochs where the analysis shall be "
                                                                   "performed.")

    parser.add_argument('--feature_dim', type=int, default=-1, help="Penultimate layer dimensionality (-1 translates to"
                                                                    "d = n_classes).")

    parser.add_argument('--data_aug', action="store_true", help="Augment data by cropping and flipping.")
    parser.add_argument('--cr_loss_weight', type=float, default=0.0)
    parser.add_argument('--cr_aug', type=str, default="autoaug_cifar10")
    parser.add_argument('--cutout', type=int, default=0)

    parser.add_argument('--sgd_momentum', type=float, default=0.9)



def init_run(args, parser, return_parser=False, create_dirs=True, return_config=False):
    if args.uid is None:
        args.uid = generate_uid(args)

    loc_config = get_config()
    base_path = get_base_path(args, loc_config)
    if not base_path.endswith('/'):
        base_path += '/'
    set_log_level(loc_config)

    if args.uid is None:
        unique_id = str(np.random.randint(0, 100000))
        logging.info("revise the unique id to a random number " + str(unique_id))
        args.uid = unique_id
        timestamp = datetime.datetime.now().strftime("%a-%b-%d-%H-%M")
        save_path = base_path + args.uid + '-' + timestamp
    else:
        save_path = base_path + str(args.uid)

    args.save_path = save_path
    args.log = save_path + '/log.txt'
    args.arg = save_path + '/args.txt'

    if not os.path.exists(save_path) and create_dirs:
        os.makedirs(save_path, exist_ok=True)
    else:
        if not args.force:
            raise ("please use another uid ")
        else:
            logging.warning("Overriding the uid '" + args.uid + "'...")
            for m in range(1, 10):
                if not os.path.exists(save_path + "/log.txt.bk" + str(m) and create_dirs):
                    shutil.copy(args.log, save_path + "/log.txt.bk" + str(m))
                    shutil.copy(args.arg, save_path + "/args.txt.bk" + str(m))
                    break

    if create_dirs:
        with open(args.log, 'w') as f:
            f.close()
        with open(args.arg, 'w') as f:
            print(args)
            print(args, file=f)
            f.close()
    if args.use_cudnn:
        logging.debug("cudnn is used")
        torch.backends.cudnn.benchmark = True
    else:
        logging.debug("cudnn is not used")
        torch.backends.cudnn.benchmark = False

    if return_parser and return_config:
        return args, parser, loc_config
    elif return_parser:
        return args, parser
    elif return_config:
        return args, loc_config
    else:
        return args


def parse_train_args(return_parser=False, create_dirs=True, return_config=False):
    parser = argparse.ArgumentParser()

    add_default_parameter(parser)

    parser.add_argument('--use_ho_uid', action='store_true', help="Use HO uid")
    parser.add_argument('--continue_ho', action='store_true', help='If true, we search for HO runs to continue')
    parser.add_argument('--cust_path', type=str, default='')
    parser.add_argument('--cust_version', type=str, default='')

    args = parser.parse_args()
    return init_run(args, parser, return_parser, create_dirs, return_config)


def parse_ho_args(return_parser=False, create_dirs=True, return_config=False):
    parser = argparse.ArgumentParser()

    add_default_parameter(parser)

    parser.add_argument('--continue_ho', action='store_true', help='If true, we search for HO runs to continue')
    parser.add_argument('--continue_ho_base_path', type=str)

    args = parser.parse_args()

    args.use_ho_uid = True
    return init_run(args, parser, return_parser, create_dirs, return_config)


def dump_args_dict(args):
    with open(args.save_path + "/args.json", 'w') as f:
        args_dict = args.__dict__

        # Remove device as it is not serializable (must not be stored as it is reconstructed from the "gpu_id" tag
        args_dict["device"] = None

        json.dump(args_dict, f, indent=2, cls=NumpyEncoder)


def load_args_dict(args_path):
    new_args = argparse.Namespace()
    with open(args_path, 'r') as f:
        new_args.__dict__ = json.load(f)

    return new_args


def compare_args(args1, args2, ignore_keys=None, filter_keys=None):
    if ignore_keys is None:
        ignore_keys = []
    if filter_keys is None:
        filter_keys = []

    args1_filtered = {k: v for k, v in args1.__dict__.items() if k not in ignore_keys and k in filter_keys}
    args2_filtered = {k: v for k, v in args2.__dict__.items() if k not in ignore_keys and k in filter_keys}
    return args1_filtered == args2_filtered
