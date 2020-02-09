#! /usr/bin/env bash
CONFIG1="rpn_r50_fpn_MaskAware_03_wc1_wr1";
CONFIG2="rpn_r50_fpn_MaskAware_04_wc1_wr1";
CONFIG3="rpn_r50_fpn_MaskAware_05_wc1_wr1";
CONFIG4="rpn_r50_fpn_MaskAware_03_wc2_wr1";
CONFIG5="rpn_r50_fpn_MaskAware_05_wc2_wr1";
python tools/test.py "configs/softIoU/MaskAwareIoU/RPN/$CONFIG1.py" "work_dirs/$CONFIG1/epoch12.pth"  —out "results/RPN/$CONFIG1/$CONFIG1.pkl" --eval proposal_fast >> "results/RPN/$CONFIG1/$CONFIG1.result";
python tools/test.py "configs/softIoU/MaskAwareIoU/RPN/$CONFIG2.py" "work_dirs/$CONFIG2/epoch12.pth"  —out "results/RPN/$CONFIG2/$CONFIG2.pkl" --eval proposal_fast >> "results/RPN/$CONFIG1/$CONFIG2.result";
python tools/test.py "configs/softIoU/MaskAwareIoU/RPN/$CONFIG3.py" "work_dirs/$CONFIG3/epoch12.pth"  —out "results/RPN/$CONFIG3/$CONFIG3.pkl" --eval proposal_fast >> "results/RPN/$CONFIG1/$CONFIG3.result";
python tools/test.py "configs/softIoU/MaskAwareIoU/RPN/$CONFIG4.py" "work_dirs/$CONFIG4/epoch12.pth"  —out "results/RPN/$CONFIG4/$CONFIG4.pkl" --eval proposal_fast >> "results/RPN/$CONFIG1/$CONFIG4.result";
python tools/test.py "configs/softIoU/MaskAwareIoU/RPN/$CONFIG5.py" "work_dirs/$CONFIG5/epoch12.pth"  —out "results/RPN/$CONFIG5/$CONFIG5.pkl" --eval proposal_fast >> "results/RPN/$CONFIG1/$CONFIG5.result";

