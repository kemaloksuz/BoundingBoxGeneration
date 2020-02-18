#! /usr/bin/env bash
CONFIG1="rpn_r101_fpn_MaskAware_05_wc1_wr2";
mkdir -p "results/RPN/$CONFIG1" &&
python tools/train.py "configs/softIoU/MaskAwareIoU/RPN/$CONFIG1.py" --gpus 4 &&
python tools/test.py "configs/softIoU/MaskAwareIoU/RPN/$CONFIG1.py" "work_dirs/$CONFIG1/epoch_12.pth"  --out "results/RPN/$CONFIG1/$CONFIG1.pkl" --eval proposal_fast >> "results/RPN/$CONFIG1/$CONFIG1.result"
