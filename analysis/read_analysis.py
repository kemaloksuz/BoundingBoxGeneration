from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import os
import pdb


if __name__ == '__main__':
    p = Path('/home/cancam/imgworkspace/mmdetection/class_analysis_exp.txt')
    out_list = []
    with p.open('rb') as f:
        #fsz = os.fstat(f.fileno()).st_size
        out = np.loadtxt(f) 
    plt.hist([out[:, 0], out[:,1]], label=["pred_labels", "gt_labels"])
    plt.yscale("log")
    plt.legend()
    plt.show()
