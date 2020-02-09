#! /usr/bin/env bash
CONFIG1="rpn_r50_fpn_MaskAware_05_wc075_wr075";
mkdir -p "results/RPN/$CONFIG1" &&
python tools/train.py "configs/softIoU/MaskAwareIoU/RPN/$CONFIG1.py" --gpus 4 &&
python tools/test.py "configs/softIoU/MaskAwareIoU/RPN/$CONFIG1.py" "work_dirs/$CONFIG1/epoch12.pth"  --out "results/RPN/$CONFIG1/$CONFIG1.pkl" --eval proposal_fast >> "results/RPN/$CONFIG1/$CONFIG1.result"
