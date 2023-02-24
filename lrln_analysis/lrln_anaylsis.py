import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import numpy as np
import pickle as pkl

from datasets import IMAGENET_MEAN, IMAGENET_STDDEV, TRAIN_DATA_FN, TRAIN_ORIG_TARGETS_FN, TRAIN_TARGETS_FN, \
    SimpleTensorDataset
from label_smoothing import BetaLabelRelaxationLoss, BetaLabelRelaxationCRLoss
from validate_NC import FCOutputs


def analysize_lrln(args, model, testloader, criterion, epoch, num_classes):
    result_dict = {}

    # Load data and labels
    data = torch.load(os.path.join(args.save_path, TRAIN_DATA_FN))
    orig_targets = torch.load(os.path.join(args.save_path, TRAIN_ORIG_TARGETS_FN))
    act_targets = torch.load(os.path.join(args.save_path, TRAIN_TARGETS_FN))

    # Create dataset and dataloader from given tensors
    train_transform = transforms.Compose([
        transforms.Grayscale(3), transforms.Resize(32), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STDDEV)
    ])
    dataset = SimpleTensorDataset(data, orig_targets, act_targets, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory)

    # Register post-softmax hook
    fc_postsoftmax = FCOutputs()
    forward_hook_handle = model.fc.register_forward_hook(fc_postsoftmax)

    model.eval()

    # Get training predictions
    train_preds = torch.zeros([len(dataloader.dataset), num_classes])
    with torch.no_grad():
        for batch_idx, (inputs, _, _) in tqdm(enumerate(dataloader)):
            inputs = inputs.to(args.device)
            _ = model(inputs)

            train_preds[batch_idx * args.batch_size:batch_idx * args.batch_size + inputs.size(0)] = \
                fc_postsoftmax.outputs[0]
            fc_postsoftmax.clear()
    train_preds = train_preds.cpu().numpy()

    # Get test predictions
    test_preds = torch.zeros([len(testloader.dataset), num_classes])
    with torch.no_grad():
        for batch_idx, (inputs, _) in tqdm(enumerate(testloader)):
            inputs = inputs.to(args.device)
            _ = model(inputs)

            test_preds[batch_idx * args.batch_size:batch_idx * args.batch_size + inputs.size(0)] = \
                fc_postsoftmax.outputs[0]
            fc_postsoftmax.clear()
    test_preds = test_preds.cpu().numpy()

    # Unregister post-softmax hook
    forward_hook_handle.remove()
    del forward_hook_handle

    orig_targets = orig_targets.cpu().numpy()
    act_targets = act_targets.cpu().numpy()

    # Masks
    clean_mask = orig_targets == act_targets
    noisy_mask = orig_targets != act_targets

    # Probabilistic analysis

    # These are the predicted magnitudes of the correct class. This is expected to be large both in the beginning and
    #   the end of the training, as the model is expected to learn to predict the correct class.
    # Calculate the mean of the clean training predictions for the actual targets
    result_dict["magnitude_clean"] = np.mean(train_preds[clean_mask][np.arange(len(train_preds[clean_mask])),
    act_targets[clean_mask]])
    result_dict["magnitude_clean_std"] = np.std(train_preds[clean_mask][np.arange(len(train_preds[clean_mask])),
    act_targets[clean_mask]])

    # These are the predicted magnitudes of the noisy labels (as they appear in the training data). I.e., this is
    #   expected to be lower in the beginning of the training, but larger eventually due to memorization.
    result_dict["magnitude_noisy"] = np.mean(train_preds[noisy_mask][np.arange(len(train_preds[noisy_mask])),
    act_targets[noisy_mask]])
    result_dict["magnitude_noisy_std"] = np.std(train_preds[noisy_mask][np.arange(len(train_preds[noisy_mask])),
    act_targets[noisy_mask]])

    # These are the predicted magnitudes of the original labels for the noisy instances. This is expected to be larger
    #   in the beginning of the training, but smaller eventually due to memorization.
    result_dict["magnitude_noisy_orig"] = np.mean(train_preds[noisy_mask][np.arange(len(train_preds[noisy_mask])),
    orig_targets[noisy_mask]])
    result_dict["magnitude_noisy_orig_std"] = np.std(train_preds[noisy_mask][np.arange(len(train_preds[noisy_mask])),
    orig_targets[noisy_mask]])

    # This is the predicted max probability for the clean labels. It is expected to mostly increase monotonically.
    result_dict["magnitude_max_pred_clean"] = np.mean(np.max(train_preds[clean_mask], axis=1))
    result_dict["magnitude_max_pred_clean_std"] = np.std(np.max(train_preds[clean_mask], axis=1))
    # This is the predicted max probability for the noisy labels. It is expected to first increase as the noisy instance
    #   is shifted towards the correct class, then decreases as it has to "move" to the other side of the decision
    #   boundary, followed by an increase due to the memorization.
    result_dict["magnitude_max_pred_noisy"] = np.mean(np.max(train_preds[noisy_mask], axis=1))
    result_dict["magnitude_max_pred_noisy_std"] = np.std(np.max(train_preds[noisy_mask], axis=1))

    result_dict["magnitude_max_pred_all"] = np.mean(np.max(train_preds, axis=1))

    non_act_orig_mask = (np.argmax(train_preds, axis=1) != orig_targets) & (
                np.argmax(train_preds, axis=1) != act_targets)
    result_dict["magnitude_non_act_orig"] = np.mean(np.max(train_preds[non_act_orig_mask], axis=-1))
    result_dict["magnitude_non_act_orig_std"] = np.std(np.max(train_preds[non_act_orig_mask], axis=-1))
    result_dict["magnitude_non_act_orig_clean"] = np.mean(np.max(train_preds[non_act_orig_mask & clean_mask], axis=-1))
    result_dict["magnitude_non_act_orig_clean_std"] = np.std(
        np.max(train_preds[non_act_orig_mask & clean_mask], axis=-1))
    result_dict["magnitude_non_act_orig_noisy"] = np.mean(np.max(train_preds[non_act_orig_mask & noisy_mask], axis=-1))
    result_dict["magnitude_non_act_orig_noisy_std"] = np.std(
        np.max(train_preds[non_act_orig_mask & noisy_mask], axis=-1))

    # Now calculate all the metrics from above for each of the classes individually
    for i in range(num_classes):
        clean_mask_act_i = clean_mask & (act_targets == i)
        noisy_mask_act_i = noisy_mask & (act_targets == i)
        clean_mask_orig_i = clean_mask & (orig_targets == i)
        noisy_mask_orig_i = noisy_mask & (orig_targets == i)

        # Actual class == i
        result_dict["magnitude_clean_{}".format(i)] = np.mean(train_preds[clean_mask_act_i][np.arange(
            len(train_preds[clean_mask_act_i])), act_targets[clean_mask_act_i]])
        result_dict["magnitude_clean_std_{}".format(i)] = np.std(train_preds[clean_mask_act_i][np.arange(
            len(train_preds[clean_mask_act_i])), act_targets[clean_mask_act_i]])
        result_dict["magnitude_noisy_{}".format(i)] = np.mean(train_preds[noisy_mask_act_i][np.arange(
            len(train_preds[noisy_mask_act_i])), act_targets[noisy_mask_act_i]])
        result_dict["magnitude_noisy_std_{}".format(i)] = np.std(train_preds[noisy_mask_act_i][np.arange(
            len(train_preds[noisy_mask_act_i])), act_targets[noisy_mask_act_i]])

        # Original class == i
        result_dict["magnitude_clean_orig_{}".format(i)] = np.mean(train_preds[clean_mask_orig_i][np.arange(
            len(train_preds[clean_mask_orig_i])), orig_targets[clean_mask_orig_i]])
        result_dict["magnitude_clean_orig_std_{}".format(i)] = np.std(train_preds[clean_mask_orig_i][np.arange(
            len(train_preds[clean_mask_orig_i])), orig_targets[clean_mask_orig_i]])
        result_dict["magnitude_noisy_orig_{}".format(i)] = np.mean(train_preds[noisy_mask_orig_i][np.arange(
            len(train_preds[noisy_mask_orig_i])), orig_targets[noisy_mask_orig_i]])
        result_dict["magnitude_noisy_orig_std_{}".format(i)] = np.std(train_preds[noisy_mask_orig_i][np.arange(
            len(train_preds[noisy_mask_orig_i])), orig_targets[noisy_mask_orig_i]])

        result_dict["magnitude_max_pred_clean_{}".format(i)] = np.mean(np.max(train_preds[clean_mask_act_i], axis=1))
        result_dict["magnitude_max_pred_clean_std_{}".format(i)] = np.std(np.max(train_preds[clean_mask_act_i], axis=1))
        result_dict["magnitude_max_pred_noisy_{}".format(i)] = np.mean(np.max(train_preds[noisy_mask_act_i], axis=1))
        result_dict["magnitude_max_pred_noisy_std_{}".format(i)] = np.std(np.max(train_preds[noisy_mask_act_i], axis=1))

        result_dict["magnitude_max_pred_clean_orig_{}".format(i)] = np.mean(
            np.max(train_preds[clean_mask_orig_i], axis=1))
        result_dict["magnitude_max_pred_clean_orig_std_{}".format(i)] = np.std(
            np.max(train_preds[clean_mask_orig_i], axis=1))
        result_dict["magnitude_max_pred_noisy_orig_{}".format(i)] = np.mean(
            np.max(train_preds[noisy_mask_orig_i], axis=1))
        result_dict["magnitude_max_pred_noisy_orig_std_{}".format(i)] = np.std(
            np.max(train_preds[noisy_mask_orig_i], axis=1))

        result_dict["magnitude_max_pred_all_{}".format(i)] = np.mean(np.max(train_preds[act_targets == i], axis=1))
        result_dict["magnitude_max_pred_all_orig_{}".format(i)] = np.mean(
            np.max(train_preds[orig_targets == i], axis=1))

    # Calculate the test magnitudes
    result_dict["test_magnitude"] = np.mean(test_preds[np.arange(len(test_preds)), testloader.dataset.targets])
    result_dict["test_magnitude_std"] = np.std(test_preds[np.arange(len(test_preds)), testloader.dataset.targets])

    # Calculate the test max probabilities
    result_dict["test_magnitude_max_pred"] = np.mean(np.max(test_preds, axis=1))
    result_dict["test_magnitude_max_pred_std"] = np.std(np.max(test_preds, axis=1))

    # Calculate the test magnitudes for correct test predictions
    result_dict["test_magnitude_correct"] = np.mean(test_preds[np.arange(len(test_preds)), testloader.dataset.targets])
    result_dict["test_magnitude_correct_std"] = np.std(
        test_preds[np.arange(len(test_preds)), testloader.dataset.targets])

    # Calculate the test magnitudes for all incorrect test predictions
    incorrect_mask = np.argmax(test_preds, axis=-1) != testloader.dataset.targets.numpy()
    result_dict["test_magnitude_incorrect"] = np.mean(np.max(test_preds, axis=1)[incorrect_mask])
    result_dict["test_magnitude_incorrect_std"] = np.std(np.max(test_preds, axis=1)[incorrect_mask])

    result_dict["test_magnitude_all"] = np.mean(np.max(test_preds, axis=1))

    # Calculate the test metrics for all classes individually
    for i in range(num_classes):
        test_mask_i = testloader.dataset.targets.numpy() == i
        result_dict["test_magnitude_{}".format(i)] = np.mean(np.max(test_preds, axis=1)[test_mask_i])
        result_dict["test_magnitude_std_{}".format(i)] = np.std(np.max(test_preds, axis=1)[test_mask_i])

        result_dict["test_magnitude_correct_{}".format(i)] = np.mean(np.max(test_preds, axis=1)[test_mask_i])
        result_dict["test_magnitude_correct_std_{}".format(i)] = np.std(np.max(test_preds, axis=1)[test_mask_i])

        result_dict["test_magnitude_incorrect_{}".format(i)] = np.mean(
            np.max(test_preds, axis=1)[incorrect_mask & test_mask_i])
        result_dict["test_magnitude_incorrect_std_{}".format(i)] = np.std(
            np.max(test_preds, axis=1)[incorrect_mask & test_mask_i])

        result_dict["test_magnitude_all_{}".format(i)] = np.mean(np.max(test_preds, axis=1)[test_mask_i])

    # Accuracy analysis
    result_dict["train_acc"] = np.mean(np.argmax(train_preds, axis=1) == act_targets)
    result_dict["train_acc_clean"] = np.mean(np.argmax(train_preds[clean_mask], axis=1) == act_targets[clean_mask])
    result_dict["train_acc_noisy"] = np.mean(np.argmax(train_preds[noisy_mask], axis=1) == act_targets[noisy_mask])

    # Get fraction of samples that are correctly classified
    result_dict["orig_fraction_correct"] = np.mean(np.argmax(train_preds, axis=1) == orig_targets)
    result_dict["orig_fraction_correct_clean"] = np.mean(
        np.argmax(train_preds[clean_mask], axis=1) == orig_targets[clean_mask])
    result_dict["orig_fraction_correct_noisy"] = np.mean(
        np.argmax(train_preds[noisy_mask], axis=1) == orig_targets[noisy_mask])
    result_dict["orig_fraction_incorrect"] = np.mean(np.argmax(train_preds, axis=1) != orig_targets)
    result_dict["orig_fraction_incorrect_clean"] = np.mean(
        np.argmax(train_preds[clean_mask], axis=1) != orig_targets[clean_mask])
    result_dict["orig_fraction_incorrect_noisy"] = np.mean(
        np.argmax(train_preds[noisy_mask], axis=1) != orig_targets[noisy_mask])

    result_dict["act_fraction_correct"] = np.mean(np.argmax(train_preds, axis=1) == act_targets)
    result_dict["act_fraction_correct_clean"] = np.mean(
        np.argmax(train_preds[clean_mask], axis=1) == act_targets[clean_mask])
    result_dict["act_fraction_correct_noisy"] = np.mean(
        np.argmax(train_preds[noisy_mask], axis=1) == act_targets[noisy_mask])
    result_dict["act_fraction_incorrect"] = np.mean(np.argmax(train_preds, axis=1) != act_targets)
    result_dict["act_fraction_incorrect_clean"] = np.mean(
        np.argmax(train_preds[clean_mask], axis=1) != act_targets[clean_mask])
    result_dict["act_fraction_incorrect_noisy"] = np.mean(
        np.argmax(train_preds[noisy_mask], axis=1) != act_targets[noisy_mask])

    # Memorization: The classification of a noisy example matches the noisy label in the actual training label data
    result_dict["memorization"] = np.mean(np.argmax(train_preds[noisy_mask], axis=1) == act_targets[noisy_mask])
    result_dict["incorrect_others"] = np.mean(
        (np.argmax(train_preds[noisy_mask], axis=1) != act_targets[noisy_mask]) & (
                np.argmax(train_preds[noisy_mask], axis=1) != orig_targets[noisy_mask]))

    # Now compute everything for each class (for the original class, see related work)
    for i in range(num_classes):
        class_mask = orig_targets == i
        class_clean_mask = class_mask & clean_mask
        class_noisy_mask = class_mask & noisy_mask

        result_dict["orig_fraction_correct_{}".format(i)] = np.mean(
            np.argmax(train_preds[class_mask], axis=1) == orig_targets[class_mask])
        result_dict["orig_fraction_correct_clean_{}".format(i)] = np.mean(
            np.argmax(train_preds[class_clean_mask], axis=1) == orig_targets[class_clean_mask])
        result_dict["orig_fraction_correct_noisy_{}".format(i)] = np.mean(
            np.argmax(train_preds[class_noisy_mask], axis=1) == orig_targets[class_noisy_mask])
        result_dict["orig_fraction_incorrect_{}".format(i)] = np.mean(
            np.argmax(train_preds[class_mask], axis=1) != orig_targets[class_mask])
        result_dict["orig_fraction_incorrect_clean_{}".format(i)] = np.mean(
            np.argmax(train_preds[class_clean_mask], axis=1) != orig_targets[class_clean_mask])
        result_dict["orig_fraction_incorrect_noisy_{}".format(i)] = np.mean(
            np.argmax(train_preds[class_noisy_mask], axis=1) != orig_targets[class_noisy_mask])

        result_dict["act_fraction_correct_{}".format(i)] = np.mean(
            np.argmax(train_preds[class_mask], axis=1) == act_targets[class_mask])
        result_dict["act_fraction_correct_clean_{}".format(i)] = np.mean(
            np.argmax(train_preds[class_clean_mask], axis=1) == act_targets[class_clean_mask])
        result_dict["act_fraction_correct_noisy_{}".format(i)] = np.mean(
            np.argmax(train_preds[class_noisy_mask], axis=1) == act_targets[class_noisy_mask])
        result_dict["act_fraction_incorrect_{}".format(i)] = np.mean(
            np.argmax(train_preds[class_mask], axis=1) != act_targets[class_mask])
        result_dict["act_fraction_incorrect_clean_{}".format(i)] = np.mean(
            np.argmax(train_preds[class_clean_mask], axis=1) != act_targets[class_clean_mask])
        result_dict["act_fraction_incorrect_noisy_{}".format(i)] = np.mean(
            np.argmax(train_preds[class_noisy_mask], axis=1) != act_targets[class_noisy_mask])

        result_dict["memorization_{}".format(i)] = np.mean(
            np.argmax(train_preds[class_noisy_mask], axis=1) == act_targets[class_noisy_mask])

    result_dict["test_acc"] = np.mean(np.argmax(test_preds, axis=1) == testloader.dataset.targets.numpy())
    result_dict["test_fraction_correct"] = np.mean(np.argmax(test_preds, axis=1) == testloader.dataset.targets.numpy())
    result_dict["test_fraction_incorrect"] = np.mean(
        np.argmax(test_preds, axis=1) != testloader.dataset.targets.numpy())

    # Credal set statistics
    if isinstance(criterion, BetaLabelRelaxationLoss) or isinstance(criterion, BetaLabelRelaxationCRLoss):
        if criterion.warmup:
            result_dict["beta"] = 1.

            result_dict["credal_set_size"] = 1.
            result_dict["credal_set_size_std"] = 0.

            result_dict["credal_set_size_clean"] = 1.
            result_dict["credal_set_size_std_clean"] = 0.

            result_dict["credal_set_size_noisy"] = 1.
            result_dict["credal_set_size_std_noisy"] = 0.

            result_dict["credal_set_size_correct"] = 1.
            result_dict["credal_set_size_std_correct"] = 0.

            result_dict["credal_set_size_incorrect"] = 1.
            result_dict["credal_set_size_std_incorrect"] = 0.

            result_dict["credal_set_size_classes"] = np.eye(num_classes)

            # Credal set validity
            result_dict["credal_set_validity"] = np.mean(act_targets == orig_targets)
            result_dict["credal_set_validity_std"] = np.std(act_targets == orig_targets)

            result_dict["credal_set_validity_clean"] = np.mean(act_targets[clean_mask] == orig_targets[clean_mask])
            result_dict["credal_set_validity_clean_std"] = np.std(act_targets[clean_mask] == orig_targets[clean_mask])

            result_dict["credal_set_validity_noisy"] = np.mean(act_targets[noisy_mask] == orig_targets[noisy_mask])
            result_dict["credal_set_validity_noisy_std"] = np.std(act_targets[noisy_mask] == orig_targets[noisy_mask])
        else:
            result_dict["beta"] = criterion.beta

            correct_mask = np.argmax(train_preds, axis=1) == orig_targets
            incorrect_mask = np.argmax(train_preds, axis=1) != orig_targets

            # One-hot targets
            one_hot_targets = np.zeros_like(train_preds)
            one_hot_targets[np.arange(len(train_preds)), act_targets] = 1.
            # To bool
            one_hot_targets = one_hot_targets.astype(bool)

            one_hot_targets_orig = np.zeros_like(train_preds)
            one_hot_targets_orig[np.arange(len(train_preds)), orig_targets] = 1.
            # To bool
            one_hot_targets_orig = one_hot_targets_orig.astype(bool)

            result_dict["credal_set_size"] = np.mean(
                np.sum(np.logical_or(train_preds > criterion.beta, one_hot_targets), axis=-1))
            result_dict["credal_set_size_std"] = np.std(
                np.sum(np.logical_or(train_preds > criterion.beta, one_hot_targets), axis=-1))

            result_dict["credal_set_size_clean"] = np.mean(
                np.sum(np.logical_or(train_preds[clean_mask] > criterion.beta, one_hot_targets[clean_mask]), axis=-1))
            result_dict["credal_set_size_clean_std"] = np.std(
                np.sum(np.logical_or(train_preds[clean_mask] > criterion.beta, one_hot_targets[clean_mask]), axis=-1))

            result_dict["credal_set_size_noisy"] = np.mean(
                np.sum(np.logical_or(train_preds[noisy_mask] > criterion.beta, one_hot_targets[noisy_mask]), axis=-1))
            result_dict["credal_set_size_noisy_std"] = np.std(
                np.sum(np.logical_or(train_preds[noisy_mask] > criterion.beta, one_hot_targets[noisy_mask]), axis=-1))

            result_dict["credal_set_size_clean_correct"] = np.mean(np.sum(
                np.logical_or(train_preds[clean_mask & correct_mask] > criterion.beta,
                              one_hot_targets[clean_mask & correct_mask]), axis=-1))
            result_dict["credal_set_size_clean_correct_std"] = np.std(np.sum(
                np.logical_or(train_preds[clean_mask & correct_mask] > criterion.beta,
                              one_hot_targets[clean_mask & correct_mask]), axis=-1))

            result_dict["credal_set_size_clean_incorrect"] = np.mean(np.sum(
                np.logical_or(train_preds[clean_mask & incorrect_mask] > criterion.beta,
                              one_hot_targets[clean_mask & incorrect_mask]), axis=-1))
            result_dict["credal_set_size_clean_incorrect_std"] = np.std(np.sum(
                np.logical_or(train_preds[clean_mask & incorrect_mask] > criterion.beta,
                              one_hot_targets[clean_mask & incorrect_mask]), axis=-1))

            credal_set_size_matrix = np.zeros((num_classes, num_classes))
            for i in range(num_classes):
                for j in range(num_classes):
                    # Mean of the credal set size for class j when we observe i as the true label
                    credal_set_size_matrix[i, j] = np.mean(np.sum(
                        np.logical_or(train_preds[orig_targets == i][j] > criterion.beta,
                                      one_hot_targets[orig_targets == i][j]), axis=-1))
            result_dict["credal_set_size_classes"] = credal_set_size_matrix

            # Credal set validity
            result_dict["credal_set_validity"] = np.mean(np.sum(np.logical_and(
                np.logical_or(train_preds > criterion.beta, one_hot_targets), one_hot_targets_orig), axis=-1))
            result_dict["credal_set_validity_std"] = np.std(np.sum(np.logical_and(
                np.logical_or(train_preds > criterion.beta, one_hot_targets), one_hot_targets_orig), axis=-1))

            result_dict["credal_set_validity_clean"] = np.mean(np.sum(np.logical_and(
                np.logical_or(train_preds[clean_mask] > criterion.beta, one_hot_targets[clean_mask]),
                one_hot_targets_orig[clean_mask]), axis=-1))
            result_dict["credal_set_validity_clean_std"] = np.std(np.sum(np.logical_and(
                np.logical_or(train_preds[clean_mask] > criterion.beta, one_hot_targets[clean_mask]),
                one_hot_targets_orig[clean_mask]), axis=-1))

            result_dict["credal_set_validity_noisy"] = np.mean(np.sum(np.logical_and(
                np.logical_or(train_preds[noisy_mask] > criterion.beta, one_hot_targets[noisy_mask]),
                one_hot_targets_orig[noisy_mask]), axis=-1))
            result_dict["credal_set_validity_noisy_std"] = np.std(np.sum(np.logical_and(
                np.logical_or(train_preds[noisy_mask] > criterion.beta, one_hot_targets[noisy_mask]),
                one_hot_targets_orig[noisy_mask]), axis=-1))

    # Save result dict to pkl
    with open(os.path.join(args.save_path, "lrln_eval_{}.pkl".format(epoch)), "wb") as f:
        pkl.dump(result_dict, f)

    # Save train and test predictions
    np.save(os.path.join(args.save_path, "lrln_train_preds_{}.npy".format(epoch)), train_preds)
    np.save(os.path.join(args.save_path, "lrln_test_preds_{}.npy".format(epoch)), test_preds)

    model.train()
