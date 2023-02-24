import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-7


class GCELoss(nn.Module):
    def __init__(self, num_classes=10, q=0.7):
        super(GCELoss, self).__init__()
        self.q = q
        self.num_classes = num_classes

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean()


class NCELoss(nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NCELoss, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = -1 * torch.sum(label_one_hot * pred, dim=1) / (-pred.sum(dim=1))
        return self.scale * loss.mean()


class AUELoss(nn.Module):
    def __init__(self, num_classes=10, a=5.5, q=3., eps=eps, scale=1.0):
        super(AUELoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.q = q
        self.eps = eps
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (torch.pow(self.a - torch.sum(label_one_hot * pred, dim=1), self.q) - (self.a - 1) ** self.q) / self.q
        return loss.mean() * self.scale


class AGCELoss(nn.Module):
    def __init__(self, num_classes=10, a=0.6, q=0.6, eps=eps, scale=1.):
        super(AGCELoss, self).__init__()
        self.a = a
        self.q = q
        self.num_classes = num_classes
        self.eps = eps
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = ((self.a + 1) ** self.q - torch.pow(self.a + torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean() * self.scale


class NCEandAGCE(torch.nn.Module):
    def __init__(self, alpha=1., beta=1., num_classes=10, a=3, q=1.5):
        super(NCEandAGCE, self).__init__()

        if num_classes == 10:
            a = 6
            q = 1.5
        elif num_classes == 100:
            a = 1.8
            q = 3
            alpha = 10
            beta = 0.1

        self.num_classes = num_classes
        self.nce = NCELoss(num_classes=num_classes, scale=alpha)
        self.agce = AGCELoss(num_classes=num_classes, a=a, q=q, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.agce(pred, labels)


class NCEandAUE(torch.nn.Module):
    def __init__(self, alpha=1., beta=1., num_classes=10, a=6, q=1.5):
        super(NCEandAUE, self).__init__()

        if num_classes == 10:
            a = 6
            q = 1.5
        elif num_classes == 100:
            a = 6
            q = 3
            alpha = 10
            beta = 0.015

        self.num_classes = num_classes
        self.nce = NCELoss(num_classes=num_classes, scale=alpha)
        self.aue = AUELoss(num_classes=num_classes, a=a, q=q, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.aue(pred, labels)
