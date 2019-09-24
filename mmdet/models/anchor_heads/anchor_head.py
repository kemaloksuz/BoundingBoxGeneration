from __future__ import division

import numpy as np
import skimage.io as io

import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import (AnchorGenerator, anchor_target, delta2bbox, force_fp32,
                        multi_apply, multiclass_nms)
from ..builder import build_loss
from ..registry import HEADS
import pdb
import inspect

from pycocotools.coco import COCO
from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt
import PIL
from PIL import Image
from scipy import ndimage

import torchvision.transforms.functional as F

@HEADS.register_module
class AnchorHead(nn.Module):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        anchor_scales (Iterable): Anchor scales.
        anchor_ratios (Iterable): Anchor aspect ratios.
        anchor_strides (Iterable): Anchor strides.
        anchor_base_sizes (Iterable): Anchor base sizes.
        target_means (Iterable): Mean values of regression targets.
        target_stds (Iterable): Std values of regression targets.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 anchor_scales=[8, 16, 32],
                 anchor_ratios=[0.5, 1.0, 2.0],
                 anchor_strides=[4, 8, 16, 32, 64],
                 anchor_base_sizes=None,
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction="none",
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, reduction="none", loss_weight=1.0)):
        super(AnchorHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(
            anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.target_means = target_means
        self.target_stds = target_stds

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss', 'GHMC']
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes - 1
        else:
            self.cls_out_channels = num_classes
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.fp16_enabled = False

        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                AnchorGenerator(anchor_base, anchor_scales, anchor_ratios))

        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)
        self._init_layers()

    def _init_layers(self):
        self.conv_cls = nn.Conv2d(self.feat_channels,
                                  self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.conv_cls, std=0.01)
        normal_init(self.conv_reg, std=0.01)

    def forward_single(self, x):
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return cls_score, bbox_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)
        return anchor_list, valid_flag_list

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, img_metas, img, matched_gt_list_, anchors_list_, num_total_samples, cfg):
        # classification loss
        pdb.set_trace()
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')
        """
        image üzerinde cls_loss'u en hatalı olan 1 tane anchor box.
        
        1. anchors_list
        2. matched_gt_list
        3. bbox2delta' -> bbox_preds [test koduna bak] 
        4. labels
        5. loss_cls (summed over dim 1)
        6. loss_bbox (summed over dim 1)
        
        how to pass image or coco object to read image??
        read coco_img with loadImgs
        read image with coco_url

        (img-meta gelecek) - check
        """
        im_2_show = plt.imread(img_metas['filename'])
        #im_2_show = np.asarray(F.to_pil_image(img.cpu()[0]))
        print("Pad shape: {}".format(img_metas['pad_shape']))
        print("Image shape: {}".format(im_2_show.shape))
        if (img_metas['flip'] == True):
            im_2_show = np.fliplr(im_2_show)    

        plt.imshow(im_2_show)
        #pdb.set_trace()
        cls_max = loss_cls.detach().cpu().numpy().sum(1).argmax()
        print("Image: {}, Cls_max: {}, Cls_val: {}, Bbox_val: {}\n"\
                .format(img_metas['filename'][-16:-4], \
                        cls_max, \
                        loss_cls.detach().cpu().numpy().sum(1)[cls_max], \
                        loss_bbox.detach().cpu().numpy().sum(1)[cls_max]))
        
        matched_gt_list_ = matched_gt_list_ / img_metas['scale_factor']
        anchors_list_ = anchors_list_ / img_metas['scale_factor']
	
        #filter anchors wrt image size
        anch_ws = anchors_list_[:,2] - anchors_list_[:,0]
        anch_hs = anchors_list_[:,3] - anchors_list_[:,1]
        anch_inds = (anch_ws * anch_hs) > (img_metas['ori_shape'][0] * img_metas['ori_shape'][1]) / 4
        anchors_list_filtered = anchors_list_[anch_inds]
        
        gt_ws = matched_gt_list_[:,2] - matched_gt_list_[:,0]
        gt_hs = matched_gt_list_[:,3] - matched_gt_list_[:,1]
        
        gt_max_ind = (gt_ws*gt_hs).argmax()
        gt_max = matched_gt_list_[gt_max_ind].cpu().numpy()
        gt_max_x = gt_max[0]
        gt_max_y = gt_max[1]
        gt_max_w = gt_max[2] - gt_max_x
        gt_max_h = gt_max[3] - gt_max_y

        rect_gt_max = Rectangle((gt_max_x, gt_max_y), gt_max_w, gt_max_h, linewidth=3	, edgecolor='b', facecolor='none')
  

        print("Gt max area: {}, Anchs max area: {}, Image Size: {}\n".format((gt_ws*gt_hs).max(),(anch_ws*anch_hs).max(), (img_metas['ori_shape'][0]*img_metas['ori_shape'][1])/16))
        print(anchors_list_filtered)

        gt_x = matched_gt_list_[cls_max].cpu().numpy()[0]
        gt_y =  matched_gt_list_[cls_max].cpu().numpy()[1]
        gt_w =  matched_gt_list_[cls_max].cpu().numpy()[2] - gt_x 
        gt_h =   matched_gt_list_[cls_max].cpu().numpy()[3] - gt_y
        gt_w = gt_w
        gt_h = gt_h

        anch_x = anchors_list_[cls_max].cpu().numpy()[0]
        anch_y = anchors_list_[cls_max].cpu().numpy()[1]
        anch_w = anchors_list_[cls_max].cpu().numpy()[2] - anch_x
        anch_h = anchors_list_[cls_max].cpu().numpy()[3] - anch_y
        anch_w = anch_w
        anch_h = anch_h
	
        ax=plt.gca()

        rect_gt = Rectangle((gt_x, gt_y), gt_w, gt_h, linewidth=3	, edgecolor='g', facecolor='none')
        rect_anch = Rectangle((anch_x, anch_y), anch_w, anch_h, linewidth=3, edgecolor='r', facecolor='none')
        
        ax.add_patch(rect_gt)
        ax.add_patch(rect_anch)
        
        ax.add_patch(rect_gt_max)
        
        ax.text(0, 0, "Loss_Cls={}\n" 
        	      "Loss_Bbox:{}\n" 
        	      "Total Loss: {}\n"
                      "Class Label: {}".format(loss_cls.detach().cpu().numpy().sum(1)[cls_max], \
        	      			      loss_bbox.detach().cpu().numpy().sum(1)[cls_max], \
        	      			      loss_cls.detach().cpu().numpy().sum(1)[cls_max]+loss_bbox.detach().cpu().numpy().sum(1)[cls_max], \
        	      			      CLASSES[labels[cls_max]-1], \
        	      			      fontsize=12))
        
        print(img_metas)
        #plt.show()
        #pdb.set_trace()
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             img,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)
        pdb.set_trace()
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        pdb.set_trace()
        cls_reg_targets = anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None
        pdb.set_trace()
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, matched_gt_list_, anchors_list_) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            img_metas,
            img,
            matched_gt_list_,
            anchors_list_,
            num_total_samples=num_total_samples,
            cfg=cfg)
        curframe = inspect.currentframe()
        callframe = inspect.getouterframes(curframe, 2)
        ##print("Processing image: {}".format(img_metas[0]['filename'][-16:-4]))
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg,
                   rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(cls_scores[i].size()[-2:],
                                                   self.anchor_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               mlvl_anchors, img_shape,
                                               scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        pdb.set_trace()
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_scores, bbox_preds,
                                                 mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = delta2bbox(anchors, bbox_pred, self.target_means,
                                self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels
