from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
import pdb

PATH = '/home/cancam/imgworkspace/mmdetection/work_dirs/lrp_optimization/faster_rcnn_r50_fpn_1x_IoU_ExpLoss_debug/class_analysis_exp.txt'

if __name__ == '__main__':
    out_exp = pd.read_csv(PATH,\
                          delim_whitespace=True,\
                          names = ["pred_labels", "gt_labels", "loss"])
    pdb.set_trace()
