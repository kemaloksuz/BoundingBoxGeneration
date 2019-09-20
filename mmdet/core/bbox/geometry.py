import torch
import numpy as np
from mmdet.core.mask.mask_target import mask_target_single

def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
            must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """

    assert mode in ['iou', 'iof']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious

def area(bboxes):
    return (bboxes[:,2]-bboxes[:,0])*(bboxes[:,3]-bboxes[:,1])

def segm_overlaps(gt_masks, gt_bboxes, bboxes, overlaps, min_overlap,plot=0): 
    import pdb

    #import time

   # start = time.time()

    segm_ious=overlaps.data.new_zeros(overlaps.size())
    #Convert list to torch

    #Find number of pixels that are in each bb
    larger_ind=overlaps>min_overlap
    nonzero_iou_ind=torch.nonzero(larger_ind)
    #gt_mask_size=torch.sum(gt_masks,dim=[1,2]).type(torch.cuda.FloatTensor)
    image_h,image_w=gt_masks[0].shape
    #end1 = time.time()
    #bboxes=bboxes.type(torch.cuda.IntTensor) 
    bboxes=torch.clamp(bboxes, min=0)
    bboxes[:,[0,2]]=torch.clamp(bboxes[:,[0,2]], max=image_w-1)
    bboxes[:,[1,3]]=torch.clamp(bboxes[:,[1,3]], max=image_h-1)

    all_boxes=bboxes[nonzero_iou_ind[:,1]]
    anchor_masks=mask_target_single(all_boxes,nonzero_iou_ind[:,0], gt_masks)
    unnorm_anchor=torch.sum(anchor_masks,dim=[1,2]).type(torch.cuda.FloatTensor)
    anchor_segm=unnorm_anchor*area(all_boxes)
    if plot:
        import matplotlib.pyplot as plt
        from matplotlib import patches as patch
        import random
        no=random.randint(0,nonzero_iou_ind.shape[0])
        pltgt,pltanc=nonzero_iou_ind[no]
        print(pltgt,pltanc)
        fig, ax = plt.subplots(1)
        ax.imshow(gt_masks[pltgt])

        tempRect=patch.Rectangle((bboxes[pltanc,0],bboxes[pltanc,1]), bboxes[pltanc,2]-bboxes[pltanc,0], bboxes[pltanc,3]-bboxes[pltanc,1],linewidth=3,edgecolor='r',facecolor='none')
        ax.add_patch(tempRect) 
        fntsize=14
        tempRect=patch.Rectangle((gt_bboxes[pltgt,0],gt_bboxes[pltgt,1]), gt_bboxes[pltgt,2]-gt_bboxes[pltgt,0], gt_bboxes[pltgt,3]-gt_bboxes[pltgt,1],linewidth=3,edgecolor='g',facecolor='none')
        ax.add_patch(tempRect)        

        ax.tick_params(labelsize=fntsize)      
        plt.xlabel('x', fontsize=fntsize)
        plt.ylabel('y', fontsize=fntsize)


    gt_number=overlaps.size()[0]
    gt_labels=torch.arange(0, gt_number).cuda()
    gt_masks=mask_target_single(gt_bboxes,gt_labels, gt_masks)
    unnorm_gt=torch.sum(gt_masks,dim=[1,2]).type(torch.cuda.FloatTensor)
    gt_segm=unnorm_gt*area(gt_bboxes)
    per_anchor_gt=gt_segm[nonzero_iou_ind[:,0]]
    segm_vals=anchor_segm/per_anchor_gt

    segm_ious[larger_ind]=segm_vals
    if plot:
        ax.text(0, 0, "iou= "+np.array2string(overlaps[pltgt,pltanc].cpu().numpy())+", "+"\n segm_iou="+np.array2string(segm_ious[pltgt,pltanc].cpu().numpy()), fontsize=12)
        plt.show()


    #end = time.time()
    #print("t=",nonzero_iou_ind.size(), end1 - start, end - start)
    return segm_ious
