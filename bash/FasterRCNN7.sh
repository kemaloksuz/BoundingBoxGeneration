#! /usr/bin/env bash
CONFIG1="faster_rcnn_r50_fpn_RPN_wc1_wr2_RCNN_wc2_wr1_w1";
mkdir -p "results/FasterRCNN/$CONFIG1"&&
python tools/train.py "configs/softIoU/MaskAwareIoU/FasterRCNN/$CONFIG1.py" --gpus 4 &&
python tools/test.py "configs/softIoU/MaskAwareIoU/FasterRCNN/$CONFIG1.py" "work_dirs/$CONFIG1/epoch_12.pth"  --json_out "results/FasterRCNN/$CONFIG1/$CONFIG1" --eval bbox &&
python tools/coco_eval.py "results/FasterRCNN/$CONFIG1/$CONFIG1.bbox.json" --ann "data/coco/annotations/instances_val2017.json" --LRPEval 1 --LRPtau 0.5 >> "results/FasterRCNN/$CONFIG1/$CONFIG1.result"
