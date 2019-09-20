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
ax.text(0, 0, "iou= "+np.array2string(overlaps[pltgt,pltanc].cpu().numpy())+", "+"\n segm_iou="+np.array2string(segm_ious[pltgt,pltanc].cpu().numpy()), fontsize=12)
plt.show()
