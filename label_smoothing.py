import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LabelSmoothingLossCanonical(nn.Module):
    def __init__(self, smoothing=0.0, dim=-1):
        super(LabelSmoothingLossCanonical, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

        print("Using Label Smoothing...")

    def forward(self, pred, target):
        # Log softmax is used for numerical stability
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist += self.smoothing / pred.size(self.dim)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class LabelRelaxationLoss(nn.Module):
    def __init__(self, alpha=0.1, dim=-1, logits_provided=True, one_hot_encode_trgts=True, num_classes=-1):
        super(LabelRelaxationLoss, self).__init__()
        self.alpha = alpha
        self.dim = dim

        # Greater zero threshold
        self.gz_threshold = 0.1

        self.logits_provided = logits_provided
        self.one_hot_encode_trgts = one_hot_encode_trgts

        self.num_classes = num_classes

    def forward(self, pred, target):
        if self.logits_provided:
            pred = pred.softmax(dim=self.dim)

        # with torch.no_grad():
        # Apply one-hot encoding to targets
        if self.one_hot_encode_trgts:
            target = F.one_hot(target, num_classes=self.num_classes)

        with torch.no_grad():
            sum_y_hat_prime = torch.sum((torch.ones_like(target) - target) * pred, dim=-1)
            pred_hat = self.alpha * pred / torch.unsqueeze(sum_y_hat_prime, dim=-1)
            target_credal = torch.where(target > self.gz_threshold, torch.ones_like(target) - self.alpha, pred_hat)
        divergence = torch.sum(F.kl_div(pred.log(), target_credal, log_target=False, reduction="none"), dim=-1)

        pred = torch.sum(pred * target, dim=-1)

        result = torch.where(torch.gt(pred, 1. - self.alpha), torch.zeros_like(divergence), divergence)
        return torch.mean(result)


class BetaLabelRelaxationLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.2, dim=-1, logits_provided=True, one_hot_encode_trgts=True, num_classes=-1,
                 adaptive_beta=False, epochs: Optional[int] = None, warmup=True, adaptive_start_beta=None,
                 adaptive_end_beta=None, adaptive_type="linear"):
        super(BetaLabelRelaxationLoss, self).__init__()
        self.alpha = max(alpha, 1e-3)
        self.dim = dim

        self.logits_provided = logits_provided
        self.one_hot_encode_trgts = one_hot_encode_trgts

        self.num_classes = num_classes

        self.beta = beta

        self.warmup = warmup
        self.adaptive_beta = adaptive_beta
        if self.adaptive_beta:
            assert epochs is not None
            self.epochs = epochs
            self.start_beta = adaptive_start_beta
            self.end_beta = adaptive_end_beta

            self.adaptive_type = adaptive_type

    def forward(self, logits, target, epoch=None):
        if self.logits_provided:
            pred_log = logits.log_softmax(dim=self.dim)
        else:
            pred_log = logits.log()

        # Apply one-hot encoding to targets
        target = F.one_hot(target, num_classes=self.num_classes)

        with torch.no_grad():
            if self.logits_provided:
                pred = logits.detach().softmax(dim=self.dim)
            else:
                pred = logits.detach()

            if self.adaptive_beta:
                if epoch is not None:
                    # beta = (1. - epoch / self.epochs) * self.max_beta
                    if self.adaptive_type == "linear":
                        beta = (1 - epoch / self.epochs) * self.start_beta + (epoch / self.epochs) * self.end_beta
                    elif self.adaptive_type == "cosine":
                        if self.start_beta < self.end_beta:
                            logging.warning("Start beta is smaller than end beta for cosine annealing.")

                        beta = self.end_beta + 0.5 * (self.start_beta - self.end_beta) * (
                                1 + math.cos(math.pi * epoch / self.epochs))
                    else:
                        raise ValueError(f"Unknown adaptive beta type: {self.adaptive_type}")
                    beta_mask = torch.logical_or(pred > beta, target)
                else:
                    # This can happen e.g. at evaluation time
                    beta_mask = target.bool()
            else:
                if self.warmup:
                    beta_mask = target.bool()
                else:
                    beta_mask = torch.logical_or(pred > self.beta, target)

            beta_sum = torch.sum(beta_mask.float() * pred, dim=-1)
            non_beta_sum = torch.sum((~beta_mask).float() * pred, dim=-1)

            beta_sum_preds = torch.where(torch.unsqueeze(beta_sum == 0., dim=-1), torch.zeros_like(pred),
                                         pred / torch.unsqueeze(beta_sum, dim=-1))
            non_beta_sum_preds = torch.where(torch.unsqueeze(non_beta_sum == 0., dim=-1), torch.zeros_like(pred),
                                             pred / torch.unsqueeze(non_beta_sum, dim=-1))

            target_credal = torch.where(beta_mask, (1. - self.alpha) * beta_sum_preds,
                                        self.alpha * non_beta_sum_preds)

        divergence = torch.sum(F.kl_div(pred_log, target_credal, log_target=False, reduction="none"), dim=-1)

        is_in_credal_set = beta_sum >= 1. - self.alpha

        result = torch.where(is_in_credal_set, torch.zeros_like(divergence), divergence)
        return torch.mean(result)


class BetaLabelRelaxationCRLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.2, one_hot_encode_trgts=True, num_classes=-1):
        super(BetaLabelRelaxationCRLoss, self).__init__()
        self.alpha = max(alpha, 1e-3)

        self.one_hot_encode_trgts = one_hot_encode_trgts

        self.num_classes = num_classes

        self.beta = beta

        self.warmup = True

    def forward(self, preds, target):
        preds_w = preds[0]
        preds_s = preds[1]

        preds1 = F.softmax(preds_w, dim=-1).detach()
        preds2 = F.log_softmax(preds_s, dim=-1)

        # with torch.no_grad():
        # Apply one-hot encoding to targets
        # if self.one_hot_encode_trgts:
        target = F.one_hot(target, num_classes=self.num_classes)

        with torch.no_grad():
            if self.warmup:
                beta_mask = target.bool()
            else:
                beta_mask = torch.logical_or(preds1 > self.beta, target)

            beta_sum = torch.sum(beta_mask.float() * preds1, dim=-1)
            non_beta_sum = torch.sum((~beta_mask).float() * preds1, dim=-1)

            beta_sum_preds = torch.where(torch.unsqueeze(beta_sum == 0., dim=-1), torch.zeros_like(preds1),
                                         preds1 / torch.unsqueeze(beta_sum, dim=-1))
            non_beta_sum_preds = torch.where(torch.unsqueeze(non_beta_sum == 0., dim=-1), torch.zeros_like(preds1),
                                             preds1 / torch.unsqueeze(non_beta_sum, dim=-1))

            target_credal = torch.where(beta_mask, (1. - self.alpha) * beta_sum_preds,
                                        self.alpha * non_beta_sum_preds)

        divergence = torch.sum(F.kl_div(preds2, target_credal, log_target=False, reduction="none"), dim=-1)

        is_in_credal_set = beta_sum >= 1. - self.alpha

        result = torch.where(is_in_credal_set, torch.zeros_like(divergence), divergence)
        return torch.mean(result)



class BetaCompleteAmbiguationLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.2, dim=-1, logits_provided=True, one_hot_encode_trgts=True, num_classes=-1,
                 adaptive_beta=False, epochs: Optional[int] = None, warmup=True, adaptive_start_beta=None,
                 adaptive_end_beta=None, adaptive_type="linear"):
        super(BetaCompleteAmbiguationLoss, self).__init__()
        self.alpha = max(alpha, 1e-3)
        self.dim = dim

        self.logits_provided = logits_provided
        self.one_hot_encode_trgts = one_hot_encode_trgts

        self.num_classes = num_classes

        self.beta = beta

        self.warmup = warmup
        self.adaptive_beta = adaptive_beta
        if self.adaptive_beta:
            assert epochs is not None
            self.epochs = epochs
            self.start_beta = adaptive_start_beta
            self.end_beta = adaptive_end_beta

            self.adaptive_type = adaptive_type

    def forward(self, logits, target, epoch=None):
        if self.logits_provided:
            pred_log = logits.log_softmax(dim=self.dim)
        else:
            pred_log = logits.log()

        # Apply one-hot encoding to targets
        target = F.one_hot(target, num_classes=self.num_classes)

        with torch.no_grad():
            if self.logits_provided:
                pred = logits.detach().softmax(dim=self.dim)
            else:
                pred = logits.detach()

            if self.adaptive_beta:
                if epoch is not None:
                    # beta = (1. - epoch / self.epochs) * self.max_beta
                    if self.adaptive_type == "linear":
                        beta = (1 - epoch / self.epochs) * self.start_beta + (epoch / self.epochs) * self.end_beta
                    elif self.adaptive_type == "cosine":
                        if self.start_beta < self.end_beta:
                            logging.warning("Start beta is smaller than end beta for cosine annealing.")

                        beta = self.end_beta + 0.5 * (self.start_beta - self.end_beta) * (
                                1 + math.cos(math.pi * epoch / self.epochs))
                    else:
                        raise ValueError(f"Unknown adaptive beta type: {self.adaptive_type}")
                    beta_mask = torch.logical_or(pred > beta, target)
                else:
                    # This can happen e.g. at evaluation time
                    beta_mask = target.bool()
            else:
                if self.warmup:
                    beta_mask = target.bool()
                else:
                    beta_mask = torch.logical_or(pred > self.beta, target)

            # If there are two true elements, then set the complete row to true
            beta_mask = torch.where(torch.unsqueeze(beta_mask.sum(-1) > 1,dim=-1), torch.ones_like(beta_mask),
                                    beta_mask)

            beta_sum = torch.sum(beta_mask.float() * pred, dim=-1)
            non_beta_sum = torch.sum((~beta_mask).float() * pred, dim=-1)

            beta_sum_preds = torch.where(torch.unsqueeze(beta_sum == 0., dim=-1), torch.zeros_like(pred),
                                         pred / torch.unsqueeze(beta_sum, dim=-1))
            non_beta_sum_preds = torch.where(torch.unsqueeze(non_beta_sum == 0., dim=-1), torch.zeros_like(pred),
                                             pred / torch.unsqueeze(non_beta_sum, dim=-1))

            target_credal = torch.where(beta_mask, (1. - self.alpha) * beta_sum_preds,
                                        self.alpha * non_beta_sum_preds)

        divergence = torch.sum(F.kl_div(pred_log, target_credal, log_target=False, reduction="none"), dim=-1)

        is_in_credal_set = beta_sum >= 1. - self.alpha

        result = torch.where(is_in_credal_set, torch.zeros_like(divergence), divergence)
        return torch.mean(result)
