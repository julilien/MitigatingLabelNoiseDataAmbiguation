import os
import pickle


def generate_uid(args):
    if args.loss == "LS":
        alpha = args.ls_alpha
    else:
        alpha = args.lr_alpha

    uid = "_".join(
        [args.dataset, str(args.label_noise), str(args.seed), args.model, args.loss, str(alpha),
         str(args.lrvar2_warmup),
         str(args.lrvar2_beta), str(args.decay_type)])

    if args.use_asymmetric_noise:
        uid += "_asym"
    if args.cr_loss_weight > 0.:
        uid += "_cr{}_{}".format(args.cr_loss_weight, args.cr_aug)
    if args.cutoff > 0:
        uid += "_cutoff{}".format(args.cutoff)

    return uid


def retrieve_model_dict(dict_id, base_path, model_name, alt_base_path=None):
    def retrieve_metrics(model_dir, model_name):
        with open(os.path.join(os.path.join(model_dir, model_name), dict_id + '.pkl'), 'rb') as f:
            return pickle.load(f)

    if alt_base_path is None:
        return retrieve_metrics(base_path, model_name)
    else:
        try:
            return retrieve_metrics(base_path, model_name)
        except FileNotFoundError:
            return retrieve_metrics(alt_base_path, model_name)
