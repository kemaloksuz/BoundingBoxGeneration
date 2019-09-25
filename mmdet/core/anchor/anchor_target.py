import torch
import pdb

from ..bbox import PseudoSampler, assign_and_sample, bbox2delta, build_assigner
from ..utils import multi_apply

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

def anchor_target(anchor_list,
                  valid_flag_list,
                  gt_bboxes_list,
                  img_metas,
                  target_means,
                  target_stds,
                  cfg,
                  gt_bboxes_ignore_list=None,
                  gt_labels_list=None,
                  label_channels=1,
                  sampling=True,
                  unmap_outputs=True):
    """Compute regression and classification targets for anchors.

    Args:
        anchor_list (list[list]): Multi level anchors of each image.
        valid_flag_list (list[list]): Multi level valid flags of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        img_metas (list[dict]): Meta info of each image.
        target_means (Iterable): Mean value of regression targets.
        target_stds (Iterable): Std value of regression targets.
        cfg (dict): RPN train configs.

    Returns:
        tuple
    """
    num_imgs = len(img_metas)
    assert len(anchor_list) == len(valid_flag_list) == num_imgs

    # anchor number of multi levels
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]    
    print("Num_level_anchors:")
    print(num_level_anchors)
    # concat all level anchors and flags to a single tensor
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])
        anchor_list[i] = torch.cat(anchor_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])
    
    # all anchors are flattened in anchor_list variable.

    # compute targets for each image
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]
    
    # select the largest anchor 
    anchor_boxes_list = anchor_list[0] / img_metas[0]['scale_factor']
    
    anchors_ws = anchor_boxes_list[:, 2] - anchor_boxes_list[:, 0]
    anchors_hs = anchor_boxes_list[:, 3] - anchor_boxes_list[:, 1]

    img_width = img_metas[0]['ori_shape'][0]
    img_height = img_metas[0]['ori_shape'][1]
    img_area = img_width * img_height
    img_area = img_area / 4

    anchor_max_ind = (anchors_ws*anchors_hs) > img_area
    anchors_max = anchor_boxes_list[anchor_max_ind].cpu().numpy()
    
    ax = plt.gca()
    print("Number of anchors after filtering: {}\n".format(anchors_max.shape[0]))     
    for anchor_single in anchors_max:
        anchors_max_x = anchor_single[0]
        anchors_max_y = anchor_single[1]
        anchors_max_w = anchor_single[2] - anchors_max_x
        anchors_max_h = anchor_single[3] - anchors_max_y
        
        rect_anchor_max = Rectangle((anchors_max_x, anchors_max_y), anchors_max_w, anchors_max_h, linewidth=3, edgecolor=np.random.rand(3,), facecolor='None')
        ax.add_patch(rect_anchor_max)

    # select the largets gt
    gt_bboxes_list[0] = gt_bboxes_list[0] / img_metas[0]['scale_factor']

    gt_ws = gt_bboxes_list[0][:,2] - gt_bboxes_list[0][:,0]
    gt_hs = gt_bboxes_list[0][:, 3] - gt_bboxes_list[0][:, 1]
    gt_max_ind = (gt_ws*gt_hs).argmax()

    gt_max = gt_bboxes_list[0][gt_max_ind].cpu().numpy()
    gt_max_x = gt_max[0]
    gt_max_y = gt_max[1]
    gt_max_w = gt_max[2] - gt_max_x
    gt_max_h = gt_max[3] - gt_max_y
    
    rect_gt_max = Rectangle((gt_max_x, gt_max_y), gt_max_w, gt_max_h, linewidth=3, edgecolor='b', facecolor='none')
    im_2_show = plt.imread(img_metas[0]['filename'])
    
    if (img_metas[0]['flip'] == True):
        im_2_show = np.fliplr(im_2_show)

    plt.imshow(im_2_show)
    ax.add_patch(rect_gt_max)
    plt.show()
    plt.gca()
    plt.gcf()


    ax.add_patch(rect_gt_max)
    pdb.set_trace()
    (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
     pos_inds_list, neg_inds_list, matched_gt_list_, anchors_list_) = multi_apply(
         anchor_target_single,
         anchor_list,
         valid_flag_list,
         gt_bboxes_list,
         gt_bboxes_ignore_list,
         gt_labels_list,
         img_metas,
         target_means=target_means,
         target_stds=target_stds,
         cfg=cfg,
         label_channels=label_channels,
         sampling=sampling,
         unmap_outputs=unmap_outputs)
    # no valid anchors
    if any([labels is None for labels in all_labels]):
        return None
    # sampled anchors of all images
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    # split targets to a list w.r.t. multiple levels
    # level seperation must be considered
    labels_list = images_to_levels(all_labels, num_level_anchors)
    label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
    bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
    matched_gt_list_ = images_to_levels(matched_gt_list_, num_level_anchors)
    anchors_list_ = images_to_levels(anchors_list_, num_level_anchors)
    
    pdb.set_trace()
    return (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, num_total_pos, num_total_neg, matched_gt_list_, anchors_list_)


def images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets


def anchor_target_single(flat_anchors,
                         valid_flags,
                         gt_bboxes,
                         gt_bboxes_ignore,
                         gt_labels,
                         img_meta,
                         target_means,
                         target_stds,
                         cfg,
                         label_channels=1,
                         sampling=True,
                         unmap_outputs=True):
    inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                       img_meta['img_shape'][:2],
                                       cfg.allowed_border)
    if not inside_flags.any():
        return (None, ) * 6
    # assign gt and sample anchors
    anchors = flat_anchors[inside_flags, :]
    print(sampling)
    if sampling:
        assign_result, sampling_result = assign_and_sample(
            anchors, gt_bboxes, gt_bboxes_ignore, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                             gt_bboxes_ignore, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors,
                                              gt_bboxes)

    num_valid_anchors = anchors.shape[0]
    bbox_targets = torch.zeros_like(anchors)
    matched_gts = torch.zeros_like(anchors)
    
    bbox_weights = torch.zeros_like(anchors)
    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
    
    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        pos_bbox_targets = bbox2delta(sampling_result.pos_bboxes,
                                      sampling_result.pos_gt_bboxes,
                                      target_means, target_stds)
        
        bbox_targets[pos_inds, :] = pos_bbox_targets
        matched_gts[pos_inds, :] = sampling_result.pos_gt_bboxes

        bbox_weights[pos_inds, :] = 1.0
        
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        
        
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    # map up to original set of anchors

    if unmap_outputs:
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        matched_gts = unmap(matched_gts, num_total_anchors, inside_flags)
        anchors = unmap(anchors, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
    return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
            neg_inds, matched_gts, anchors)


def anchor_inside_flags(flat_anchors, valid_flags, img_shape,
                        allowed_border=0):
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
            (flat_anchors[:, 0] >= -allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 1] >= -allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 2] < img_w + allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 3] < img_h + allowed_border).type(torch.uint8)
    else:
        inside_flags = valid_flags
    return inside_flags


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret
