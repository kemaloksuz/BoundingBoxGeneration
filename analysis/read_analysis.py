from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
import pdb


if __name__ == '__main__':
    
    p = Path('/home/cancam/imgworkspace/mmdetection/analysis/class_analysis_exp.txt')
    p_write = Path('/home/cancam/imgworkspace/mmdetection/analysis/class_analysis_exp_.txt')

    with p.open('rb') as f:
        out_exp = np.loadtxt(f)

    with p_write.open('wb') as fp:
        np.savetxt(fp, out_exp, fmt='%4.6f')
        fp.close()

    p = Path('/home/cancam/imgworkspace/mmdetection/analysis/class_analysis_ce.txt')
    p_write = Path('/home/cancam/imgworkspace/mmdetection/analysis/class_analysis_ce_.txt')
    with p.open('rb') as f:
        out_ce = np.loadtxt(f)
    
    with p_write.open('wb') as fp:
        np.savetxt(fp, out_ce, fmt='%4.6f')
        fp.close()
    
    pdb.set_trace()
    out_exp = pd.read_csv("/home/cancam/imgworkspace/mmdetection/analysis/class_analysis_exp_.txt",\
                          delim_whitespace=True, \
                          names=["pred_labels_exp","gt_labels_exp","loss_exp"])

    out_ce = pd.read_csv("/home/cancam/imgworkspace/mmdetection/analysis/class_analysis_ce_.txt",\
                          delim_whitespace=True, \
                          names=["pred_labels_ce","gt_labels_ce","loss_ce"]) 
    pdb.set_trace()
    
    hist_exp_pred = np.histogram(out_exp['pred_labels_exp'])
    hist_exp_gt = np.histogram(out_exp['gt_labels_exp'])
    
    hist_ce_pred = np.histogram(out_ce['pred_labels_ce'])
    hist_ce_gt = np.histogram(out_ce['gt_labels_ce'])
   
    pdb.set_trace()
    print("Prediction labels stats.")
    print("Exp BG:{}".format(hist_exp_pred[0]))
    print("Exp FG:{}".format(hist_exp_pred[1:].sum()))
    print("CE BG:{}".format(hist_ce_pred[0]))
    print("CE FG:{}".format(hist_ce_pred[1:].sum()))
    
    print("GT labels stats.")
    print("Exp BG:{}".format(hist_exp_gt[0]))
    print("Exp FG:{}".format(hist_exp_gt[1:].sum()))
    print("CE BG:{}".format(hist_ce_gt[0]))
    print("CE FG:{}".format(hist_ce_gt[1:].sum()))

    plt.rcParams["figure.figsize"] = (20,5)
    fig, ax = plt.subplots(2, 1, sharey=True, sharex=False)
    # histogramda gt ve pred arasında yüksek fark olanların loss değerlerine bak,
        # daha iyi anlamak için az arada yüksek fark barındıran örneklerin
        # gt_prob değerleri consider edilmeli, bunu nasıl toplayabiliriz incele.
    _, _, p0 = ax[0].hist([out_ce['gt_labels_ce'], out_ce['pred_labels_ce']], \
             bins = np.arange(0,82,1), \
             color = ['g', 'r'], \
             label = ["GT Labels", "Prediction Labels"], \
             alpha=0.75, \
             align='left')
    ax[0].grid(alpha=0.75)
    ax[0].set_title('Cross Entropy')
    ax[0].legend()
    _, _, p1 = ax[1].hist([out_exp['gt_labels_exp'], out_exp['pred_labels_exp']], \
             bins = np.arange(0,82,1), \
             color = ['g', 'r'], \
             alpha = 0.75, \
             align = 'left', \
             )
    ax[1].grid(alpha=0.75)
    ax[1].set_title('Exponential Loss')
    fig.subplots_adjust(hspace=0.3)
    plt.yscale("log")
    plt.show()
