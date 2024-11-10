import torch
import torch.nn as nn
import torch.nn.functional as F


def mask2rgb(mask):
    color_dict = {0: torch.tensor([0, 0, 0]),
                  1: torch.tensor([1, 0, 0]),
                  2: torch.tensor([0, 1, 0])}
    output = torch.zeros((mask.shape[0], mask.shape[1], mask.shape[2], 3)).long()
    for i in range(mask.shape[0]):
        for k in color_dict.keys():
            output[i][mask[i].long() == k] = color_dict[k]
    return output.to(mask.device)


@torch.no_grad()
def dice_score(
    inputs: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    # compute softmax over the classes axis
    input_one_hot = mask2rgb(inputs.argmax(dim=1))

    # create the labels one hot tensor
    target_one_hot = mask2rgb(targets)

    # compute the actual dice score
    dims = (2, 3)
    intersection = torch.sum(input_one_hot * target_one_hot, dims)
    cardinality = torch.sum(input_one_hot + target_one_hot, dims)

    dice_score = (2. * intersection + 1e-6) / (cardinality + 1e-6)
    return dice_score.mean()


class DiceLoss(nn.Module):
    def __init__(self, weights=torch.Tensor([[0.4, 0.55, 0.05]])) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6
        self.weights: torch.Tensor = weights

    def forward(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor) -> torch.Tensor:
        # compute softmax over the classes axis
        input_soft = F.softmax(inputs, dim=1)

        # create the labels one hot tensor
        target_one_hot = mask2rgb(targets)

        # compute the actual dice score
        dims = (2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)

        dice_score = torch.sum(
            dice_score * self.weights.to(dice_score.device),
            dim=1
        )
        return torch.mean(1. - dice_score)


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=1):

        # compute softmax over the classes axis
        input_soft = F.softmax(inputs, dim=1)

        # create the labels one hot tensor
        target_one_hot = mask2rgb(targets)
        # flatten label and prediction tensors
        input_soft = input_soft.view(-1)
        target_one_hot = target_one_hot.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (input_soft * target_one_hot).sum()
        FP = ((1-target_one_hot) * input_soft).sum()
        FN = (target_one_hot * (1-input_soft)).sum()

        tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
        focal_tversky = (1 - tversky)**gamma

        return focal_tversky