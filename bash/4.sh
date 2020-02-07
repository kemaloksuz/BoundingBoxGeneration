#! /usr/bin/env bash
CONFIG1="rpn_r50_fpn_MaskAware_03_wc2_wr1";
CONFIG2="rpn_r50_fpn_MaskAware_04_wc2_wr1";
mkdir -p "results/RPN/$CONFIG1"&&\
python tools/train.py "configs/softIoU/MaskAwareIoU/RPN/$CONFIG1.py" --gpus 4&&\
python tools/test.py "configs/softIoU/MaskAwareIoU/RPN/$CONFIG1.py" "work_dirs/configs/softIoU/MaskAwareIoU/RPN/$CONFIG1.py/epoch12.pth"  —out "results/RPN/$CONFIG1/$CONFIG1.pkl" --eval proposal_fast >> "results/RPN/$CONFIG1/$CONFIG1.result"&&\
mkdir -p "results/RPN/$CONFIG2"&&\
python tools/train.py "configs/softIoU/MaskAwareIoU/RPN/$CONFIG2.py" --gpus 4&&\
python tools/test.py "configs/softIoU/MaskAwareIoU/RPN/$CONFIG2.py" "work_dirs/configs/softIoU/MaskAwareIoU/RPN/$CONFIG2.py/epoch12.pth"  —out "results/RPN/$CONFIG2/$CONFIG2.pkl" --eval proposal_fast >> "results/RPN/$CONFIG2/$CONFIG2.result"
