import sys, math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss, _WeightedLoss

sys.path.append("utils")
from datasets import ChartDataset


def dice_coefficient(input, target, smooth=1.0):
    """input : is a torch variable of size BatchxnclassesxHxW representing
    log probabilities for each class
    target : is a 1-hot representation of the groundtruth, shoud have same size
    as the input"""

    assert input.size() == target.size(), 'Input sizes must be equal.'
    assert input.dim() == 4, 'Input must be a 4D Tensor.'
    uniques = np.unique(target.data.cpu().numpy())
    assert set(list(uniques)) <= set(
        [0, 1]), 'Target must only contain zeros and ones.'
    assert smooth > 0, 'Smooth must be greater than 0.'

    probs = F.softmax(input, dim=1)
    target_f = target.float()

    num = probs * target_f         # b, c, h, w -- p*g
    num = torch.sum(num, dim=3)    # b, c, h
    num = torch.sum(num, dim=2)    # b, c

    den1 = probs * probs           # b, c, h, w -- p^2
    den1 = torch.sum(den1, dim=3)  # b, c, h
    den1 = torch.sum(den1, dim=2)  # b, c

    den2 = target_f * target_f     # b, c, h, w -- g^2
    den2 = torch.sum(den2, dim=3)  # b, c, h
    den2 = torch.sum(den2, dim=2)  # b, c

    dice = (2 * num + smooth) / (den1 + den2 + smooth)

    return dice


def dice_loss(input, target, optimize_bg=False, weight=None,
              smooth=1.0, size_average=True, reduce=True):
    """input : is a torch variable of size BatchxnclassesxHxW representing
    log probabilities for each class
    target : is a 1-hot representation of the groundtruth, shoud have same size
    as the input

    weight (Variable, optional): a manual rescaling weight given to each
            class. If given, has to be a Variable of size "nclasses"""

    dice = dice_coefficient(input, target, smooth=smooth)
    size_average = size_average  # https://github.com/ashafaei/OD-test/issues/2

    if not optimize_bg:
        # we ignore bg dice val, and take the fg
        dice = dice[:, 1:]

    if not isinstance(weight, type(None)):
        if not optimize_bg:
            weight = weight[1:]             # ignore bg weight
        weight = weight.size(0) * weight / weight.sum()  # normalize fg weights
        dice = dice * weight      # weighting

    # loss is calculated using mean over fg dice vals
    dice_loss = 1 - dice.mean(1)

    if not reduce:
        return dice_loss

    if size_average:
        return dice_loss.mean()

    return dice_loss.sum()


class DiceLoss(_WeightedLoss):

    def __init__(self, optimize_bg=False, weight=None,
                 smooth=1.0, size_average=True, reduce=True):
        """input : is a torch variable of size BatchxnclassesxHxW representing
        log probabilities for each class
        target : is a 1-hot representation of the groundtruth, shoud have same
        size as the input

        weight (Variable, optional): a manual rescaling weight given to each
                class. If given, has to be a Variable of size "nclasses"""

        super(DiceLoss, self).__init__(weight, size_average)
        self.optimize_bg = optimize_bg
        self.smooth = smooth
        self.reduce = reduce
        self.size_average = size_average  # https://github.com/ashafaei/OD-test/issues/2

    def forward(self, input, target):
        target.requires_grad = False
        return dice_loss(input, target, optimize_bg=self.optimize_bg,
                         weight=self.weight, smooth=self.smooth,
                         size_average=self.size_average,
                         reduce=self.reduce)


class DiceCoefficient(torch.nn.Module):

    def __init__(self, smooth=1.0):
        """input : is a torch variable of size BatchxnclassesxHxW representing
        log probabilities for each class
        target : is a 1-hot representation of the groundtruth, shoud have same
        size as the input"""
        super(DiceCoefficient, self).__init__()

        self.smooth = smooth

    def forward(self, input, target):
        target.requires_grad = False
        return dice_coefficient(input, target, smooth=self.smooth)


def calculate_means(pred, gt, n_objects, max_n_objects, usegpu):
    """Calculate mean embedding

       pred: (bs, height * width, n_filters) - (N, 256*256, 32)
       gt: (bs, height * width, n_instances) - (N, 256*256, 20)"""

    bs, n_loc, n_filters = pred.size()
    n_instances = gt.size(2)

    pred_repeated = pred.unsqueeze(2).expand(
        bs, n_loc, n_instances, n_filters)  # (bs, n_loc, n_instances, n_filters) - (N, 256*256, 20, 32)
    gt_expanded = gt.unsqueeze(3)  # (bs, n_loc, n_instances, 1) - (N, 256*256, 20, 1) : 0 (b.g) or 1 (instance area)

    # extract pixel values from each embedding space which belong to leaf instance area
    pred_masked = pred_repeated * gt_expanded  # (N, 256*256, 20, 32)

    means = []
    for i in range(bs):
        _n_objects_sample = n_objects[i]
        # (n_loc, n_objects, n_filters)
        _pred_masked_sample = pred_masked[i, :, : _n_objects_sample]  # (256*256, n_objects, 32)
        _gt_expanded_sample = gt_expanded[i, :, : _n_objects_sample]  # (256*256, n_objects, 1)

        # calculate mean embedding for each instance
        _mean_sample = _pred_masked_sample.sum(0) / _gt_expanded_sample.sum(0)  # (n_objects, 32)
        if (max_n_objects - _n_objects_sample) != 0:
            n_fill_objects = int(max_n_objects - _n_objects_sample)
            _fill_sample = torch.zeros(n_fill_objects, n_filters)
            if usegpu:
                _fill_sample = _fill_sample.cuda()
            _fill_sample = Variable(_fill_sample)
            _mean_sample = torch.cat((_mean_sample, _fill_sample), dim=0)  # (20, 32)
        means.append(_mean_sample)

    means = torch.stack(means)  # (N, 20, 32)

    # means = pred_masked.sum(1) / gt_expanded.sum(1)
    # # bs, n_instances, n_filters

    return means


def calculate_variance_term(pred, gt, means, n_objects, delta_v, norm=2):
    """An intra-cluster pull-force that draws embeddings towards the mean embedding

       pred: bs, height * width, n_filters
       gt: bs, height * width, n_instances
       means: bs, n_instances, n_filters"""

    bs, n_loc, n_filters = pred.size()  # (N, 256*256, 32)
    n_instances = gt.size(2)

    # bs, n_loc, n_instances, n_filters
    means = means.unsqueeze(1).expand(bs, n_loc, n_instances, n_filters)  # (N, 256*256, 20, 32)
    # bs, n_loc, n_instances, n_filters
    pred = pred.unsqueeze(2).expand(bs, n_loc, n_instances, n_filters)  # (N, 256*256, 20, 32)
    # bs, n_loc, n_instances, n_filters
    gt = gt.unsqueeze(3).expand(bs, n_loc, n_instances, n_filters)  # (N, 256*256, 20, 32)

    _var = (torch.clamp(torch.norm((pred - means), norm, 3) -
                        delta_v, min=0.0) ** 2) * gt[:, :, :, 0]  # (N, 256*256, 20)

    var_term = 0.0
    for i in range(bs):
        _var_sample = _var[i, :, :n_objects[i]]  # n_loc, n_objects
        _gt_sample = gt[i, :, :n_objects[i], 0]  # n_loc, n_objects

        var_term += torch.sum(_var_sample) / torch.sum(_gt_sample)
    var_term = var_term / bs

    return var_term


def calculate_distance_term(means, n_objects, delta_d, norm=2, usegpu=True):
    """An inter-cluster push-force that pushes clusters away from each other,
    increasing the distance btw the cluster centers

    means: bs, n_instances, n_filters"""

    bs, n_instances, n_filters = means.size()

    dist_term = 0.0
    for i in range(bs):
        _n_objects_sample = int(n_objects[i])

        if _n_objects_sample <= 1:
            continue

        _mean_sample = means[i, : _n_objects_sample, :]  # n_objects, n_filters
        means_1 = _mean_sample.unsqueeze(1).expand(
            _n_objects_sample, _n_objects_sample, n_filters)
        means_2 = means_1.permute(1, 0, 2)

        diff = means_1 - means_2  # n_objects, n_objects, n_filters

        _norm = torch.norm(diff, norm, 2)

        margin = 2 * delta_d * (1.0 - torch.eye(_n_objects_sample))
        if usegpu:
            margin = margin.cuda()
        margin = Variable(margin)

        _dist_term_sample = torch.sum(
            torch.clamp(margin - _norm, min=0.0) ** 2)
        _dist_term_sample = _dist_term_sample / \
            (_n_objects_sample * (_n_objects_sample - 1))
        dist_term += _dist_term_sample

    dist_term = dist_term / bs

    return dist_term


def calculate_regularization_term(means, n_objects, norm):
    """A small pull force that draws all clusters towards the origin

    means: bs, n_instances, n_filters"""

    bs, n_instances, n_filters = means.size()

    reg_term = 0.0
    for i in range(bs):
        _mean_sample = means[i, : n_objects[i], :]  # n_objects, n_filters
        _norm = torch.norm(_mean_sample, norm, 1)
        reg_term += torch.mean(_norm)
    reg_term = reg_term / bs

    return reg_term


def discriminative_loss(input, target, n_objects,
    max_n_objects=10, delta_v=0.5, delta_d=1.5, norm=2, usegpu=True):
    """input: (N, embedding_dims, h, w) - instance segmentation prediction (N, 32, 256, 256)
       target: (N, n_instances, h, w) - GT (N, 20, 256, 256), 20=MAX_N_OBJECTS
                * 20 are not actual # of instances, if actual # of instances are smaller than 20, remainders are filled with zeros
       n_objects: bs - # of clusters(instances) of GT = actual # of instances
       max_n_objects - pre-defined at settings/data_setting.py"""

    alpha = beta = 1.0
    gamma = 0.001

    bs, n_filters, height, width = input.size()
    n_instances = target.size(1)  # =max_n_objects

    input = input.permute(0, 2, 3, 1).contiguous().view(
        bs, height * width, n_filters)  # (N, 256*256, 32)
    target = target.permute(0, 2, 3, 1).contiguous().view(
        bs, height * width, n_instances)  # (N, 256*256, 20)

    cluster_means = calculate_means(
        input, target, n_objects, max_n_objects, usegpu)

    var_term = calculate_variance_term(
        input, target, cluster_means, n_objects, delta_v, norm)
    dist_term = calculate_distance_term(
        cluster_means, n_objects, delta_d, norm, usegpu)
    reg_term = calculate_regularization_term(cluster_means, n_objects, norm)

    loss = alpha * var_term + beta * dist_term + gamma * reg_term

    return loss


class DiscriminativeLoss(_Loss):

    def __init__(self, delta_var=0.5, delta_dist=1.5, norm=2,
                 size_average=True, reduce=True, usegpu=True):
        super(DiscriminativeLoss, self).__init__(size_average)
        self.reduce = reduce
        self.size_average=True  # https://github.com/ashafaei/OD-test/issues/2

        assert self.size_average
        assert self.reduce

        self.delta_var = float(delta_var)
        self.delta_dist = float(delta_dist)
        self.norm = int(norm)
        self.usegpu = usegpu

        assert self.norm in [1, 2]

    def forward(self, input, target, n_objects, max_n_objects=10):
        target.requires_grad = False
        return discriminative_loss(input, target, n_objects, max_n_objects,
                                   self.delta_var, self.delta_dist, self.norm,
                                   self.usegpu)


class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_index)
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        return super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs)


class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_index=-1, thresh=0.7, min_kept=100000, use_weight=True, **kwargs):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                                        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                                        1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred, target):
        n, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = prob.transpose(0, 1).reshape(c, -1)

        if self.min_kept > num_valid:
            print("Lables: {}".format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(1 - valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
            kept_mask = mask_prob.le(threshold)
            valid_mask = valid_mask * kept_mask
            target = target * kept_mask.long()

        target = target.masked_fill_(1 - valid_mask, self.ignore_index)
        target = target.view(n, h, w)

        return self.criterion(pred, target)


class MixSoftmaxCrossEntropyOHEMLoss(OhemCrossEntropy2d):
    def __init__(self, aux_weight=0.4, weight=None, min_kept=100000, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyOHEMLoss, self).__init__(min_kept=min_kept, ignore_index=ignore_index)
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        return super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(*inputs)
        # return self.forward(*inputs)


# dice BCE loss
class DiceBCELoss(nn.Module):
    def __init__(self, device):
        super(DiceBCELoss, self).__init__()
        self.device = device
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, target):
        smooth = 1.
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
        BCE = nn.BCELoss()
        # cast to float
        pred = pred.float()
        target = target.float()
        pred = self.sigmoid(pred)
        return BCE(pred, target) + loss.mean()


# # dice loss
# class DiceLoss(nn.Module):
#     def __init__(self, device):
#         super(DiceLoss, self).__init__()
#         self.device = device

#     def forward(self, pred, target):
#         smooth = 1.
#         pred = pred.contiguous()
#         target = target.contiguous()
#         intersection = (pred * target).sum(dim=2).sum(dim=2)
#         loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
#         return loss.mean()


class DetectorLoss(nn.Module):
    def __init__(self, device):    
        super(DetectorLoss, self).__init__()
        self.device = device

    def bbox_iou(self, box1, box2, eps=1e-7):
        # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
        box1 = box1.t()
        box2 = box2.t()

        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps
        iou = inter / union

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height

        # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
        s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
        s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
        sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
        sin_alpha_1 = torch.abs(s_cw) / sigma
        sin_alpha_2 = torch.abs(s_ch) / sigma
        threshold = pow(2, 0.5) / 2
        sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
        rho_x = (s_cw / cw) ** 2
        rho_y = (s_ch / ch) ** 2
        gamma = angle_cost - 2
        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
        iou = iou - 0.5 * (distance_cost + shape_cost)

        return iou
        
    def build_target(self, preds, targets):
        N, C, H, W = preds.shape
        gt_box, gt_cls, ps_index = [], [], []
        quadrant = torch.tensor([[0, 0], [1, 0], 
                                 [0, 1], [1, 1]], device=self.device)

        if targets.shape[0] > 0:
            scale = torch.ones(6).to(self.device)
            scale[2:] = torch.tensor(preds.shape)[[3, 2, 3, 2]]
            gt = targets * scale
            gt = gt.repeat(4, 1, 1)
            quadrant = quadrant.repeat(gt.size(1), 1, 1).permute(1, 0, 2)
            gij = gt[..., 2:4].long() + quadrant
            j = torch.where(gij < H, gij, 0).min(dim=-1)[0] > 0 

            gi, gj = gij[j].T
            batch_index = gt[..., 0].long()[j]
            ps_index.append((batch_index, gi, gj))

            gbox = gt[..., 2:][j]
            gt_box.append(gbox)
            gt_cls.append(gt[..., 1].long()[j])

        return gt_box, gt_cls, ps_index

        
    def forward(self, preds, targets):
        ft = torch.cuda.FloatTensor if preds[0].is_cuda else torch.Tensor
        cls_loss, iou_loss, obj_loss = ft([0]), ft([0]), ft([0])
        # seg loss
        seg_loss = ft([0])

        BCEcls = nn.NLLLoss()
        BCEobj = nn.SmoothL1Loss(reduction='none')
        # seg, dice loss
        BCEseg = DiceLoss(self.device)

        pred_det, pred_seg = preds
        target_det, target_seg = targets

        gt_box, gt_cls, ps_index = self.build_target(pred_det, target_det)

        pred = pred_det.permute(0, 2, 3, 1)
        pobj = pred[:, :, :, 0]
        preg = pred[:, :, :, 1:5]
        pcls = pred[:, :, :, 5:]

        N, H, W, C = pred.shape
        tobj = torch.zeros_like(pobj) 
        factor = torch.ones_like(pobj) * 0.75

        if len(gt_box) > 0:
            b, gx, gy = ps_index[0]
            ptbox = torch.ones((preg[b, gy, gx].shape)).to(self.device)
            ptbox[:, 0] = preg[b, gy, gx][:, 0].tanh() + gx
            ptbox[:, 1] = preg[b, gy, gx][:, 1].tanh() + gy
            ptbox[:, 2] = preg[b, gy, gx][:, 2].sigmoid() * W
            ptbox[:, 3] = preg[b, gy, gx][:, 3].sigmoid() * H

            iou = self.bbox_iou(ptbox, gt_box[0])
            # Filter
            f = iou > iou.mean()
            b, gy, gx = b[f], gy[f], gx[f]

            iou = iou[f]
            iou_loss = (1.0 - iou).mean()

            ps = torch.log(pcls[b, gy, gx])
            cls_loss = BCEcls(ps, gt_cls[0][f])

            # iou aware
            tobj[b, gy, gx] = iou.float()
            n = torch.bincount(b)
            factor[b, gy, gx] =  (1. / (n[b] / (H * W))) * 0.25
        
        ### seg loss whatever box is detected or not
        seg_loss = BCEseg(pred_seg, target_seg).mean()

        obj_loss = (BCEobj(pobj, tobj) * factor).mean()

        loss = (iou_loss * 1.6) + (obj_loss * 3.2) + cls_loss * 0.1 + seg_loss * 128
        return iou_loss, obj_loss, cls_loss, seg_loss, loss


# main
if __name__ == "__main__":
    device = torch.device("cpu")
    loss = DetectorLoss(device=device)
    dataset = torch.utils.data.DataLoader(
        ChartDataset("configs/train_line.txt", aug=True),
        batch_size=1)
    model = Detector(10, 10, False)
    for img, label, mask in dataset:
        img = img.to(device).float() / 255.0
        y = model(img)
        iou, obj, cls, seg, total = loss(y, (label, mask))
        print("\niou loss: ", iou)
        print("obj loss: ", obj)
        print("cls loss: ", cls)
        print("seg loss: ", seg)
        print("total loss: ", total)        
        break
