import numpy as np
import torch

# # If fg_num_rois>32
# # Get number of unique classes
# labelsImage=labels[i][fg_inds]
# classesImage=labelsImage.cpu().unique().cuda()
# uniqueClassNum=classesImage.numel()
# # Create fg_num_rois sized array with all probs 1/unique_classes
# probs=np.ones(fg_num_rois)/uniqueClassNum
# # Find each positive RoI Number From Each Class with indices and normalize probs
# eachClassInstanceNum=np.zeros(uniqueClassNum)
# for cl in range(uniqueClassNum):
#   #print(labelsImage)
#   #print(classesImage)
#   #print(labelsImage==classesImage[cl])
#   #print(labelsImage[labelsImage==classesImage[cl]])
#   idx=(labelsImage==classesImage[cl])
#   eachClassInstanceNum[cl]=labelsImage[idx].numel()
#   probs[idx.nonzero()]/=eachClassInstanceNum[cl]
# # Sample according to probs
# fg_inds=torch.from_numpy(np.random.choice(fg_inds, fg_rois_per_image, False, probs)).cuda()

class ForegroundBalancedPosSampler(RandomSampler):

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        pos_inds = torch.nonzero(assign_result.gt_inds > 0)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            # Get unique classes and find length
            unique_classes = assign_result.labels[pos_inds].unique()
            num_classes = len(unique_gt_inds)
            # Create fg_num_rois sized array with all probs 1/unique_classes            
            probs=np.ones(pos_inds.numel())/num_classes
            # Find each positive RoI Number From Each Class with indices and normalize probs            
            for i in unique_classes:
                classes_inds = torch.nonzero(assign_result.labels == i.item())
                index=(pos_inds==classes_inds)
                probs[index]/=index.numel()
            # Sample according to probs  
            sampled_inds=torch.from_numpy(np.random.choice(pos_inds, num_expected, False, probs)).cuda()                          
            return sampled_inds
