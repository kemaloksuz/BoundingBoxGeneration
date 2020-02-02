import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from matplotlib import pyplot as plt
from pathlib import Path
from ..registry import LOSSES
from .utils import weight_reduce_loss
import pdb

def get_valid_labels(valid_labels,
                    valid_preds,
                    loss):
    tuple_ = np.concatenate((np.expand_dims(valid_preds.detach().cpu().numpy(), 1), \
                            np.expand_dims(valid_labels.detach().cpu().numpy(), 1), \
                            np.expand_dims(loss.detach().cpu().numpy(), 1)), \
                            axis = 1)
    p = Path('/home/cancam/imgworkspace/mmdetection/work_dirs/lrp_optimization/faster_rcnn_r50_fpn_1x_1-IoU_ExpLoss01_31/class_analysis_exp.txt')
    with p.open("ab") as fp:
        np.savetxt(fp, tuple_)
        fp.close()


def lrp_loss(pred, 
             label, 
             weight=None, 
             reduction='mean', 
             avg_factor=None,
             use_modulator = False,
             gamma = 1.0,
             eps = 1e-6):
    pred_softmax = F.softmax(pred)
    #pred_softmax_ = pred_softmax[:, label]
    valid_inds = ((weight>0).nonzero()).flatten()
    valid_labels = label[valid_inds]
    valid_preds = pred_softmax[valid_inds, valid_labels]
    pred_labels = pred_softmax[valid_inds, :].argmax(dim=1)
    if weight is not None:
        weight = weight.float()
    
<<<<<<< HEAD
    #loss = torch.cos(1.57*valid_preds+1.57)+1
    loss = torch.exp((-valid_preds)/0.3)    
    if use_modulator:
        loss = torch.pow((1-valid_preds), gamma)*loss

=======
    loss = torch.exp(-1*valid_preds/gamma)
    get_valid_labels(valid_labels, pred_labels, loss)
>>>>>>> b399a832a1d6477798d43ecb5e93cc0ba38a90b4
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)
    return loss


def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels>=1).squeeze()

    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
                label_weights.size(0), label_channels)

    return bin_labels, bin_label_weights


@LOSSES.register_module
class LRPLoss(nn.Module):

    def __init__(self,
                 use_sigmoid = False,
                 use_mask = False,
                 reduction = 'mean',
                 use_modulator = False,
                 gamma = 1.0,
                 loss_weight=1.0):
        super(LRPLoss, self).__init__()
        
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.use_modulator = use_modulator
        self.gamma = gamma
        
    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * lrp_loss(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            use_modulator = self.use_modulator,
            gamma = self.gamma,
            eps = 1e-6,
            **kwargs)
        return loss_cls

