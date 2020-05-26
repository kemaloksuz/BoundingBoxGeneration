import torch.nn as nn
import torch
from mmdet.core import bbox2result, bbox_mapping_back, multiclass_nms
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector

import pdb

@DETECTORS.register_module
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
		      gt_masks=None):
        #print(img_metas) 
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore,gt_masks=gt_masks)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        nms = True
        bbox_inputs = outs + (img_metas, self.test_cfg, nms, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def merge_aug_results(self, aug_bboxes, aug_scores, img_metas):
        
        recovered_bboxes = []
        
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip=img_info[0]['flip']
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        
        if aug_scores is None:
            return bboxes
        else:
            scores=torch.cat(aug_scores, dim=0)
            return bboxes, scores
    
    
    def aug_test(self, imgs, img_metas, rescale=False):

        aug_bboxes = []
        aug_scores = []
        nms = False
        for img, img_meta in zip(imgs, img_metas):
            x = self.extract_feat(img)
            outs = self.bbox_head(x)
            bbox_inputs = outs + (img_meta, self.test_cfg, nms, False)
            det_bboxes, det_scores = self.bbox_head.get_bboxes(*bbox_inputs)[0]
            aug_bboxes.append(det_bboxes)
            aug_scores.append(det_scores)
        
        merged_bboxes, merged_scores = self.merge_aug_results(aug_bboxes, aug_scores, img_metas)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,\
                                                self.test_cfg.score_thr,
                                                self.test_cfg.nms,
                                                self.test_cfg.max_per_img)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] = det_bboxes.new_tensor(
                    img_metas[0][0]['scale_factor'])
        
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                    self.bbox_head.num_classes)
        return bbox_results
