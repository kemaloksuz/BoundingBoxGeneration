import torch

from .assign_result import AssignResult
from .base_assigner import BaseAssigner
from .fullgrad.fullgrad import FullGrad
import torchvision.transforms as transforms
from mmcv.parallel import MMDataParallel
import mmcv
import torchvision.models as models
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import os.path as osp
import pdb


class MaxSemanticIoUAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.

    - -1: don't care
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
    """
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
    
    CLASSES_sorted = np.sort(CLASSES)

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 model_path,
                 img_path,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 b=0,
                 im_scale=224):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.img_path = img_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        
        self.model = self.init_model(model_path).to(self.device)       
        self.model = MMDataParallel(self.init_model(model_path), \
                                        device_ids=range(2)).cuda()
        self.fullgrad = FullGrad(self.model, self.device)
        self.im_scale=im_scale
        self.b=b

    def load_checkpoint(self, fpath):
        r"""Loads checkpoint.
        ``UnicodeDecodeError`` can be well handled, which means
        python2-saved files can be read from python3.
        Args:
            fpath (str): path to checkpoint.
        Returns:
            dict
        Examples::  
            >>> from torchreid.utils import load_checkpoint
            >>> fpath = 'log/my_model/model.pth.tar-10'
            >>> checkpoint = load_checkpoint(fpath)"""
        if fpath is None:
            raise ValueError('File path is None')
        if not osp.exists(fpath):
            raise FileNotFoundError('File is not found at "{}"'.format(fpath))
        map_location = None if torch.cuda.is_available() else 'cpu'
        try:
            checkpoint = torch.load(fpath, map_location=map_location)
        except UnicodeDecodeError:
            pickle.load = partial(pickle.load, encoding="latin1")
            pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
            checkpoint = torch.load(fpath, pickle_module=pickle, map_location=map_location)
        except Exception:
            print('Unable to load checkpoint from "{}"'.format(fpath))
            raise
        return checkpoint

    def load_pretrained_weights(self, model, weight_path):
        checkpoint = self.load_checkpoint(weight_path)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    
        model_dict = model.state_dict()
        new_state_dict = OrderedDict()
        matched_layers, discarded_layers = [], []
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:] # discard module.

            if k in model_dict and model_dict[k].size() == v.size():
                new_state_dict[k] = v
                matched_layers.append(k)
            else:
                discarded_layers.append(k)

        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict)
        if len(matched_layers) == 0:
            warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(weight_path))
        else:
            print('Successfully loaded pretrained weights from "{}"'.format(weight_path))
            if len(discarded_layers) > 0:
                print('** The following layers are discarded '
                  'due to unmatched keys or layer size: {}'.format(discarded_layers))
        return model       

    def init_model(self, path):
        num_classes = 80
        model = models.vgg16_bn(pretrained = False)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        model=self.load_pretrained_weights(model, path)
        for child in model.children():
            if not isinstance(child, torch.nn.modules.pooling.AdaptiveAvgPool2d):
                for i in range(len(child)):
                    if isinstance(child[i], torch.nn.modules.activation.ReLU):
                        child[i] = torch.nn.ReLU(inplace=False)
        return model

    def integral_image_compute(self, masks, gt_number, h, w, device):
        integral_images= [None] * gt_number
        pad_row=torch.zeros([gt_number,1,w]).type(torch.DoubleTensor).to(device)
        pad_col=torch.zeros([gt_number,h+1,1]).type(torch.DoubleTensor).to(device)
        integral_images=torch.cumsum(torch.cumsum(torch.cat([pad_col,torch.cat([pad_row,masks],dim=1)], dim=2),dim=1), dim=2)
        return integral_images

    def integral_image_fetch(self, mask, bboxes):
        bboxes[:,[2,3]]+=1
        TLx=bboxes[:,0].tolist()
        TLy=bboxes[:,1].tolist()
        BRx=bboxes[:,2].tolist()
        BRy=bboxes[:,3].tolist()
        area=mask[BRy,BRx]+mask[TLy,TLx]-mask[TLy,BRx]-mask[BRy,TLx]
        return area

    def convert_boxes(self, box):
        area = (box[0], box[1], box[2], box[3])
        return area

    def get_transforms(self, part):
        return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ])

    def get_indices(self, labels, CLASSES, CLASSES_sorted):
        labels = labels - 1 
        names = np.take(CLASSES, labels.cpu().numpy())
        indices_ = []
        for name in names:
            indices_.append(np.where(CLASSES_sorted == name)[0][0])
        return np.array(indices_)

    def normalize_saliency_map(self, saliency_map):
        saliency_map = saliency_map - torch.min(saliency_map)
        saliency_map = saliency_map / torch.max(saliency_map)
        saliency_map = (saliency_map / torch.sum(saliency_map))*(self.size*self.size)
        return torch.clip(saliency_map,min=0,max=1)

    def tensor2imgs(self, tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
        num_imgs = tensor.size(0)
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        imgs = []
        for img_id in range(num_imgs):
            img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
            img = mmcv.imdenormalize(
                img, mean, std, to_bgr=to_rgb).astype(np.uint8)
            imgs.append(np.ascontiguousarray(img))
        return imgs

    def seperate_parts(self, img_tensor, bboxes, labels, img_meta, debug=True):
        partwlabel = []
        integral_list = []

        # convert labels to sorted labels
        labels_ = self.get_indices(labels, self.CLASSES, self.CLASSES_sorted)

        for idx, box in enumerate(bboxes):
            #!!!!!!!NEED TO CROP CORRECTLY!!!!
            pdb.set_trace()
            img = self.tensor2imgs(img_tensor, **img_meta[0]['img_norm_cfg'])
            part = transforms.functional.crop(img, box[0], box[1], box[2]- box[0], box[3]- box[1])

            # transform images
            transforms_fun = self.get_transforms(part)
            if part.mode!='RGB':
                part=part.convert('RGB')

            part_ = transforms_fun(part).to(self.device)
            part_ = part_.unsqueeze(0) 
                
            # iterate over converted labels
            if debug:
                with torch.no_grad():
                    self.model.eval()
                    raw_output = self.model(part_)
                    probs = torch.softmax(raw_output[0], dim=0)
#                    part_inversed = misc_functions.get_transforms_inverse(part_[0,:,:,:].cpu())
#                    saliency_map = misc_functions.save_saliency_map(part_inversed, cam[0,:,:,:], \
#                                                                './dummy.jpg')
            cam = self.fullgrad.saliency(part_, target_class=torch.tensor([labels_[idx]]))
            saliency_map = self.normalize_saliency_map(cam[0,:,:,:])            

            integral_saliency_map = self.integral_image_compute(saliency_map, 1, self.size, self.size, device = self.device).squeeze()

            integral_list.append(integral_saliency_map)        
            if debug:
                part.show()
                print(self.CLASSES[labels[idx]-1])
            
        return integral_list


    def find_gt_scaling_factors(gt_bboxes):
        scaling_factor_x=(gt_bboxes[:, 2]-gt_bboxes[:, 0])/self.im_scale
        scaling_factor_y=(gt_bboxes[:, 3]-gt_bboxes[:, 1])/self.im_scale
        return scaling_factor_x, scaling_factor_y

    def saliency_aware_bbox_overlaps(self,gt_bboxes, bboxes, gt_labels, img, img_meta):
        rows = gt_bboxes.size(0)
        cols = bboxes.size(0)
        _, gt_saliency_integral_maps = self.seperate_parts(img, gt_bboxes, gt_labels, img_meta)

        with torch.no_grad():
            saliency_aware_overlap=gt_bboxes.data.new_zeros(rows,cols)
            lt = torch.max(gt_bboxes[:, None, :2], bboxes[:, :2])  # [rows, cols, 2]
            rb = torch.min(gt_bboxes[:, None, 2:], bboxes[:, 2:])  # [rows, cols, 2]
            scaling_factor_x, scaling_factor_y=self.find_gt_scaling_factors(gt_bboxes)            
            for i in range(rows):
                lt[i,:,:]-= gt_bboxes[i, :2]
                rb[i,:,:]-= gt_bboxes[i, :2]
                lt[i,:,0]/=scaling_factor_x
                rb[i,:,0]/=scaling_factor_x
                lt[i,:,1]/=scaling_factor_y
                rb[i,:,1]/=scaling_factor_y  
                intersected_regions=torch.concat([lt[i,:,:],rb[i,:,:]], axis=2)
                #Compute Saliency Map
                saliency_aware_overlap[i,:]=self.integral_image_fetch(gt_saliency_integral_maps[i], intersected_regions)



            area_detections=((normalized_detections[:, 2] - normalized_detections[:, 0] + 1) * (
                    normalized_detections[:, 3] - normalized_detections[:, 1] + 1))/(scaling_factor_x*scaling_factor_y)


            ious = saliency_aware_overlap / ((self.im_scale*self.im_scale) + area_detections - saliency_aware_overlap)

        if plot:
            import matplotlib.pyplot as plt
            from matplotlib import patches as patch
            import random
            larger_ind=overlaps>min_overlap
            nonzero_iou_ind=torch.nonzero(larger_ind)
            #gt_mask_size=torch.sum(gt_masks,dim=[1,2]).type(torch.cuda.FloatTensor)
            #end1 = time.time()
            #bboxes=bboxes.type(torch.cuda.IntTensor) 
            bboxes=torch.clamp(bboxes, min=0)
            bboxes[:,[0,2]]=torch.clamp(bboxes[:,[0,2]], max=image_w-1)
            bboxes[:,[1,3]]=torch.clamp(bboxes[:,[1,3]], max=image_h-1)

            no=random.randint(0,nonzero_iou_ind.shape[0])
            pltgt,pltanc=nonzero_iou_ind[no]
           # print(pltgt,pltanc)
            fig, ax = plt.subplots(1)
            ax.imshow(gt_masks[pltgt].cpu().numpy())

            tempRect=patch.Rectangle((bboxes[pltanc,0],bboxes[pltanc,1]), bboxes[pltanc,2]-bboxes[pltanc,0], bboxes[pltanc,3]-bboxes[pltanc,1],linewidth=3,edgecolor='r',facecolor='none')
            ax.add_patch(tempRect) 
            fntsize=14
            tempRect=patch.Rectangle((gt_bboxes[pltgt,0],gt_bboxes[pltgt,1]), gt_bboxes[pltgt,2]-gt_bboxes[pltgt,0], gt_bboxes[pltgt,3]-gt_bboxes[pltgt,1],linewidth=3,edgecolor='g',facecolor='none')
            ax.add_patch(tempRect)        

            ax.tick_params(labelsize=fntsize)      
            plt.xlabel('x', fontsize=fntsize)
            plt.ylabel('y', fontsize=fntsize)
            ax.text(0, 0, "iou= "+np.array2string(overlaps[pltgt,pltanc].cpu().numpy())+", "+\
                "\n segm_rate="+np.array2string(segm_ious[pltgt,pltanc].cpu().numpy())+", "+\
                "\n soft_iou="+np.array2string(soft_ious[pltgt,pltanc].cpu().numpy()), fontsize=12)
            plt.show()

        return ious    

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None, img=None, img_meta=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, 0, or a positive number. -1 means don't care,
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to -1
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        if bboxes.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            raise ValueError('No gt or bboxes')
        bboxes = bboxes[:, :4]
        overlaps = self.saliency_aware_bbox_overlaps(gt_bboxes, bboxes, gt_labels, img, img_meta)

        if (self.ignore_iof_thr > 0) and (gt_bboxes_ignore is not None) and (
                gt_bboxes_ignore.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = bbox_overlaps(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = bbox_overlaps(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        return assign_result

    def assign_wrt_overlaps(self, overlaps, gt_labels=None):
        """Assign w.r.t. the overlaps of bboxes with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        if overlaps.numel() == 0:
            raise ValueError('No gt or proposals')

        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # 2. assign negative: below
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                             & (max_overlaps < self.neg_iou_thr[1])] = 0

        # 3. assign positive: above positive IoU threshold
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        # 4. assign fg: for each gt, proposals with highest IoU
        for i in range(num_gts):
            if gt_max_overlaps[i] >= self.min_pos_iou:
                if self.gt_max_assign_all:
                    max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                    assigned_gt_inds[max_iou_inds] = i + 1
                else:
                    assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_bboxes, ))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
