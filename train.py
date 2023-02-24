import models
from analysis import dynamics_analysis
from lrln_analysis.lrln_anaylsis import analysize_lrln
from models import SimpleMLP
from utils.utils import *
from args import parse_train_args, dump_args_dict
from datasets import make_selected_dataset, AugmentedDataset
from viz.eval_simple_model_viz import evaluate_model_visually
from viz.metrics import update_accuracy

import logging

from torchsummary import summary


def assess_loss(args, model, criterion, outputs, targets, is_binary, num_classes=-1, **kwargs):
    if args.loss in [CROSS_ENTROPY_TAG, LABEL_SMOOTHING_TAG, LABEL_RELAXATION_TAG, BETA_LABEL_RELAXATION_TAG,
                     BETA_LABEL_RELAXATION_CR_TAG, GCE_TAG, NCE_TAG, AUE_TAG, AGCE_TAG, NCEAUE_TAG, NCEAGCE_TAG,
                     BETA_COMPLETE_AMB_TAG]:
        if is_binary:
            loss = criterion(torch.squeeze(outputs), torch.squeeze(targets).float(), **kwargs)
        else:
            loss = criterion(outputs, targets, **kwargs)
    elif args.loss == MSE_TAG:
        loss = criterion(outputs,
                         nn.functional.one_hot(targets, num_classes=num_classes).type(torch.FloatTensor).to(
                             args.device))
    else:
        raise NotImplementedError("No routine for loss {} implemented.".format(args.loss))

    # Now decide whether to add weight decay on last weights and last features
    if args.sep_decay:
        # Find features and weights
        features = outputs[1]
        w = model.fc.weight
        b = model.fc.bias
        lamb = args.weight_decay / 2
        lamb_feature = args.feature_decay_rate / 2
        loss += lamb * (torch.sum(w ** 2) + torch.sum(b ** 2)) + lamb_feature * torch.sum(features ** 2)

    return loss


def analyse_dynamics(args):
    return "trace_dynamics" in args and args.trace_dynamics


def evaluate_dataloader(dataloader, model, args, criterion, logfile, is_binary, prefix="val", print_stats=True,
                        step_id=None, num_classes=-1, return_top1=False):
    losses_am = AverageMeter()
    top1_am = AverageMeter()

    # Calculate metrics
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        model.eval()
        outputs = model(inputs)

        val_loss = assess_loss(args, model, criterion, outputs, targets, is_binary, num_classes=num_classes)

        update_accuracy(top1_am, inputs, outputs, targets, is_binary)
        losses_am.update(val_loss.item(), inputs.size(0))

    if print_stats:
        if step_id is not None:
            stat_str = '[epoch: %d] {}_loss: %.4f | {}_top1: %.4f '.format(prefix, prefix) % (
                step_id + 1, losses_am.avg, top1_am.avg)
        else:
            stat_str = '{}_loss: %.4f | {}_top1: %.4f '.format(prefix, prefix) % (
                losses_am.avg, top1_am.avg)
        print_and_save(stat_str, logfile)

    if return_top1:
        return top1_am.avg


def trainer(args, model, trainloader, valloader, epoch_id, criterion, optimizer, scheduler, logfile, is_binary,
            num_classes):
    losses = AverageMeter()
    top1 = AverageMeter()
    grad_norm = AverageMeter()

    kwargs = {}
    if args.adaptive_lrvar2:
        kwargs["epoch"] = epoch_id

    print_and_save('\nTraining Epoch: [%d | %d] LR: %f' % (epoch_id + 1, args.epochs, scheduler.get_last_lr()[-1]),
                   logfile)
    multi_input = isinstance(trainloader.dataset, AugmentedDataset)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if not multi_input:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
        else:
            inputs, inputs_s = inputs
            inputs, inputs_s, targets = inputs.to(args.device), inputs_s.to(args.device), targets.to(args.device)

        model.train()
        outputs = model(inputs)
        if multi_input:
            outputs_s = model(inputs_s)
            outputs = (outputs, outputs_s)

        loss = assess_loss(args, model, criterion, outputs, targets, is_binary, num_classes=num_classes, **kwargs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        grads = [p.grad.data.norm(2)for p in list(filter(lambda p: p.grad is not None, model.parameters()))]
        grad_norm.update(sum(grads).item(), inputs.size(0))

        # measure accuracy and record loss
        model.eval()
        outputs = model(inputs)
        update_accuracy(top1, inputs, outputs, targets, is_binary)

        losses.update(loss.item(), inputs.size(0))

        if batch_idx % 10 == 0:
            print_and_save('[epoch: %d] (%d/%d) | Loss: %.4f | top1: %.4f | grads: %.4f' %
                           (epoch_id + 1, batch_idx + 1, len(trainloader), losses.avg, top1.avg, grad_norm.avg), logfile)

    scheduler.step()

    if valloader is not None:
        evaluate_dataloader(valloader, model, args, criterion, logfile, is_binary, prefix="val", step_id=epoch_id,
                            num_classes=num_classes)


def train(args, model, trainloader, valloader, testloader, num_classes, val_prefix="test", force_retrain=False):
    is_binary = False

    criterion = make_criterion(args, num_classes, is_binary=is_binary)
    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer)

    logfile = open('%s/train_log.txt' % (args.save_path), 'w')

    if os.path.exists(
            os.path.join(args.save_path, "epoch_" + str(args.epochs).zfill(3) + ".pth")) and not args.force_retrain \
            and not force_retrain:
        logging.info("Model already exists, loading this model...")
        model.load_state_dict(torch.load(os.path.join(args.save_path, "epoch_" + str(args.epochs).zfill(3) + ".pth")))
    else:
        print_and_save('# of model parameters: ' + str(count_network_parameters(model)), logfile)
        print_and_save('--------------------- Training -------------------------------', logfile)
        for epoch_id in range(args.epochs):
            trainer(args, model, trainloader, valloader, epoch_id, criterion, optimizer, scheduler, logfile, is_binary,
                    num_classes)

            # Save last model
            if (epoch_id + 1) % args.epochs == 0:
                torch.save(model.state_dict(), args.save_path + "/epoch_" + str(epoch_id + 1).zfill(3) + ".pth")

            if analyse_dynamics(args) and (epoch_id + 1) % args.trace_freq == 0:
                # Trace dynamics
                # torch.save(model.state_dict(), args.save_path + "/epoch_" + str(epoch_id + 1).zfill(3) + ".pth")
                dynamics_analysis(args, model, trainloader, testloader, num_classes, epoch_id)

            analysize_lrln(args, model, testloader, criterion, epoch_id, num_classes)

            if isinstance(criterion, BetaLabelRelaxationLoss) and args.lrvar2_warmup > 0 and \
                    epoch_id == (args.lrvar2_warmup - 1):
                print_and_save("## Finished warmup. ##", logfile)
                criterion.warmup = False

    test_val_acc = evaluate_dataloader(testloader, model, args, criterion, logfile, is_binary, prefix=val_prefix,
                                       num_classes=num_classes, return_top1=True)

    logfile.close()

    return test_val_acc


def main():
    args = parse_train_args()
    if args.val_split_prop == 0.0:
        args.val_split_prop = None

    set_seed(manualSeed=args.seed)

    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    args.device = device

    if args.classes == 2:
        selected_labels = [0, 1]
    else:
        selected_labels = [i for i in range(args.classes)]

    cr_aug = args.cr_aug if (args.cr_loss_weight > 0 or args.loss == BETA_LABEL_RELAXATION_CR_TAG) else None
    trainloader, valloader, testloader, num_classes = make_selected_dataset(args, val_split_prop=args.val_split_prop,
                                                                            label_noise=args.label_noise,
                                                                            selected_labels=selected_labels,
                                                                            num_workers=args.num_workers,
                                                                            pin_memory=args.pin_memory,
                                                                            use_asymmetric_noise=args.use_asymmetric_noise,
                                                                            cr_aug=cr_aug, cutout=args.cutout)
    logging.debug("Training data #: {}".format(len(trainloader)))
    if valloader is not None:
        logging.debug("Validation data #: {}".format(len(valloader)))
    logging.debug("Test data #: {}".format(len(testloader)))

    if args.model in ["SimpleMLP", "MLP"]:
        if "feature_dim" not in args.__dict__.keys() or args.feature_dim == -1:
            num_penultimate_features = num_classes
        else:
            num_penultimate_features = args.feature_dim

        model = SimpleMLP(hidden=args.width, depth=args.depth, fc_bias=args.bias, num_classes=num_classes,
                          penultimate_layer_features=num_penultimate_features, final_activation=args.act_fn,
                          use_bn=args.use_bn, use_layer_norm=args.use_layer_norm).to(device)
    elif args.model.lower().startswith("resnet"):
        num_penultimate_features = 512
        model = models.__dict__[args.model](num_classes=num_classes, fc_bias=args.bias, ETF_fc=args.ETF_fc,
                                            fixdim=args.fixdim, SOTA=args.SOTA).to(device)
    else:
        raise ValueError("Non supported model {}.".format(args.model))

    summary(model, input_size=(3, 32, 32), batch_size=1)

    train(args, model, trainloader, valloader, testloader, num_classes)

    if num_penultimate_features <= 2:
        evaluate_model_visually(args, model, base_path=args.save_path)

    dump_args_dict(args)


if __name__ == "__main__":
    main()
