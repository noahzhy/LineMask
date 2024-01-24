"""Evaluation Metrics for Semantic Segmentation"""
import torch
import numpy as np
from torch import Tensor

__all__ = ['SegmentationMetric', 'batch_pix_accuracy', 'batch_intersection_union',
           'pixelAccuracy', 'intersectionAndUnion', 'hist_info', 'compute_score']


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
            # convert to one-hot format
            mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)


class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass):
        super(SegmentationMetric, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """

        def evaluate_worker(self, pred, label):
            correct, labeled = batch_pix_accuracy(pred, label)
            inter, union = batch_intersection_union(pred, label, self.nclass)

            self.total_correct += correct
            self.total_label += labeled
            if self.total_inter.device != inter.device:
                self.total_inter = self.total_inter.to(inter.device)
                self.total_union = self.total_union.to(union.device)
            self.total_inter += inter
            self.total_union += union

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, preds, labels)
        elif isinstance(preds, (list, tuple)):
            for (pred, label) in zip(preds, labels):
                evaluate_worker(self, pred, label)

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
        pixAcc = 1.0 * self.total_correct / (2.220446049250313e-16 + self.total_label)  # remove np.spacing(1)
        IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)
        mIoU = IoU.mean().item()
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = torch.zeros(self.nclass)
        self.total_union = torch.zeros(self.nclass)
        self.total_correct = 0
        self.total_label = 0


def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are numpy array, output 4D, target 4D
    predict = torch.argmax(output, 1) + 1
    target = torch.argmax(target, 1) + 1
    pixel_labeled = torch.sum(target > 0)
    pixel_correct = torch.sum((predict == target) * (target > 0))
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 4D
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1
    target = torch.argmax(target, 1).float() + 1

    predict = predict.float() * (target > 0).float()
    intersection = predict * (predict == target).float()
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)
    area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)
    area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)
    area_union = area_pred + area_lab - area_inter
    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
    return area_inter.float(), area_union.float()


def pixelAccuracy(imPred, imLab):
    """
    This function takes the prediction and label of a single image, returns pixel-wise accuracy
    To compute over many images do:
    for i = range(Nimages):
         (pixel_accuracy[i], pixel_correct[i], pixel_labeled[i]) = \
            pixelAccuracy(imPred[i], imLab[i])
    mean_pixel_accuracy = 1.0 * np.sum(pixel_correct) / (np.spacing(1) + np.sum(pixel_labeled))
    """
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(imLab >= 0)
    pixel_correct = np.sum((imPred == imLab) * (imLab >= 0))
    pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return (pixel_accuracy, pixel_correct, pixel_labeled)


def intersectionAndUnion(imPred, imLab, numClass):
    """
    This function takes the prediction and label of a single image,
    returns intersection and union areas for each class
    To compute over many images do:
    for i in range(Nimages):
        (area_intersection[:,i], area_union[:,i]) = intersectionAndUnion(imPred[i], imLab[i])
    IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1)+area_union, axis=1)
    """
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab >= 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection
    return (area_intersection, area_union)


def hist_info(pred, label, num_cls):
    assert pred.shape == label.shape
    k = (label >= 0) & (label < num_cls)
    labeled = np.sum(k)
    correct = np.sum((pred[k] == label[k]))

    return np.bincount(num_cls * label[k].astype(int) + pred[k], minlength=num_cls ** 2).reshape(num_cls,
                                                                                                 num_cls), labeled, correct


def compute_score(hist, correct, labeled):
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    freq = hist.sum(1) / hist.sum()
    freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
    mean_pixel_acc = correct / labeled

    return iu, mean_IU, mean_IU_no_back, mean_pixel_acc
