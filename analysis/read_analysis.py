from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
import pdb


PATH = '/home/cancam/imgworkspace/mmdetection-test/mmdetection/work_dirs/lrp_optimization/samplerExperiments/'
EXP = 'faster_rcnn_r50_fpn_1x_1-IoU_ExpLoss005_21/'
EXP_CE = 'faster_rcnn_r50_fpn_1x_1-IoUx2_CE/'
FILE = PATH+EXP+'class_analysis_exp.txt'
FILE_ = PATH+EXP+'class_analysis_exp_.txt'
FILE_CE = PATH+EXP_CE+'class_analysis_exp_.txt'
FIG = PATH+EXP+'loss_analysis_exp_.pdf'

if __name__ == '__main__':
    p = Path(FILE)
    p_write = Path(FILE_)

    with p.open('rb') as f:
        out_exp = np.loadtxt(f)

    with p_write.open('ab') as fp:
        np.savetxt(fp, out_exp, fmt='%4.6f')
        fp.close()
    
    #p = Path('/home/cancam/imgworkspace/mmdetection/analysis/class_analysis_ce.txt')
    #p_write = Path('/home/cancam/imgworkspace/mmdetection/analysis/class_analysis_ce_.txt')
    #with p.open('rb') as f:
    #    out_ce = np.loadtxt(f)
    
    #with p_write.open('ab') as fp:
    #    np.savetxt(fp, out_ce, fmt='%4.6f')
    #    fp.close()
    
    out_exp = pd.read_csv(FILE_,\
                          delim_whitespace=True, \
                          names=["pred_labels_exp","gt_labels_exp","loss_exp"])

    #out_ce = pd.read_csv(FILE_CE,\
    #                      delim_whitespace=True, \
    #                      names=["pred_labels_exp","gt_labels_exp","loss_exp"])
    
    bg_idx = np.where(out_exp["gt_labels_exp"] == 0)[0]
    fg_idx = np.where(out_exp["gt_labels_exp"] != 0)[0]
    loss_bg_exp = out_exp["loss_exp"][bg_idx]
    loss_fg_exp = out_exp["loss_exp"][fg_idx]
    
    #bg_idx_ce = np.where(out_ce["gt_labels_exp"] == 0)[0]
    #fg_idx_ce = np.where(out_ce["gt_labels_exp"] != 0)[0]
    #loss_bg_ce = out_ce["loss_exp"][bg_idx]
    #loss_fg_ce = out_ce["loss_exp"][fg_idx]
    
    colors = ['r', 'g']

    plt.hist([loss_bg_exp, loss_fg_exp], \
              label=['loss_bg_exp', 'loss_fg_exp'], \
	      align = 'left', \
              color = colors)
    plt.grid()
    plt.legend()
    plt.xlabel('Loss bins')
    plt.ylabel('Count')
    plt.title('Exponential Loss Distribution')
    plt.yscale("log")
    plt.savefig(FIG,\
                dpi=2000,\
                format='pdf')

    #plt.hist([loss_bg_ce, loss_fg_ce], 
    #         label=['loss_bg_ce', 'loss_fg_ce'],
    #         align = 'left', \
    #	     color = colors)
    #plt.grid()
    #plt.legend()
    #plt.xlabel('Loss bins')
    #plt.ylabel('Count')
    #plt.title('CE Loss Distribution')
    #plt.yscale("log")
    #plt.show()

    pdb.set_trace()

    #out_ce = pd.read_csv("/home/cancam/imgworkspace/mmdetection/analysis/class_analysis_ce_.txt",\
    #                      delim_whitespace=True, \
    #                      names=["pred_labels_ce","gt_labels_ce","loss_ce"]) 
    
    hist_exp_pred = np.histogram(out_exp['pred_labels_exp'])[0]
    hist_exp_gt = np.histogram(out_exp['gt_labels_exp'])[0]
    
    #hist_ce_pred = np.histogram(out_ce['pred_labels_ce'])[0]
    #hist_ce_gt = np.histogram(out_ce['gt_labels_ce'])[0]
     
    print("Prediction labels stats.")
    print("Exp BG:{}".format(hist_exp_pred[0] / hist_exp_pred.sum()*100))
    print("Exp FG:{}".format(hist_exp_pred[1:].sum() / hist_exp_pred.sum()*100))
    #print("CE BG:{}".format(hist_ce_pred[0] / hist_ce_pred.sum()*100))
    #print("CE FG:{}".format(hist_ce_pred[1:].sum() / hist_ce_pred.sum()*100)) 

    print("GT labels stats.")
    print("Exp BG:{}".format(hist_exp_gt[0] / hist_exp_gt.sum()*100))
    print("Exp FG:{}\n".format(hist_exp_gt[1:].sum() / hist_exp_gt.sum()*100))
    #print("CE BG:{}".format(hist_ce_gt[0] / hist_ce_gt.sum()*100))
    #print("CE FG:{}".format(hist_ce_gt[1:].sum() / hist_ce_gt.sum()*100))

    print("TOTALS:")
    print("Exp fg total: {}".format(hist_exp_pred[1:].sum()))
    print("Exp bg total: {}".format(hist_exp_pred[0]))
    print("Exp DET total: {}".format(hist_exp_pred.sum()))
    #print("CE fg total: {}".format(hist_ce_pred[1:].sum()))
    #print("CE bg total: {}".format(hist_ce_pred[0]))
    #print("CE DET total: {}".format(hist_ce_pred.sum()))

   
    #pdb.set_trace()

    plt.rcParams["figure.figsize"] = (20,5)
    fig, ax = plt.subplots(1, 1, sharey=True, sharex=False)
    # histogramda gt ve pred arasında yüksek fark olanların loss değerlerine bak,
        # daha iyi anlamak için az arada yüksek fark barındıran örneklerin
        # gt_prob değerleri consider edilmeli, bunu nasıl toplayabiliriz incele.
    #_, _, p0 = ax[0].hist([out_ce['gt_labels_ce'], out_ce['pred_labels_ce']], \
    #         bins = np.arange(0,82,1), \
    #         color = ['g', 'r'], \
    #         label = ["GT Labels", "Prediction Labels"], \
    #         alpha=0.75, \
    #         align='left')
    #ax[0].grid(alpha=0.75)
    #ax[0].set_title('Cross Entropy')
    #ax[0].legend()
    _, _, p1 = ax.hist([out_exp['gt_labels_exp'], out_exp['pred_labels_exp']], \
             bins = np.arange(0,82,1), \
             color = ['g', 'r'], \
             alpha = 0.75, \
             align = 'left', \
             )
    ax.grid(alpha=0.75)
    ax.set_title('Exponential Loss')
    fig.subplots_adjust(hspace=0.3)
    plt.yscale("log")
    plt.savefig(FIG,\
                dpi=2000,\
                format='pdf')
