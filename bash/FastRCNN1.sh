#! /usr/bin/env bash
CONFIG1="fast_rcnn_r50_fpn";
mkdir -p "results/FastRCNN/$CONFIG1"&&
python tools/train.py "configs/softIoU/MaskAwareIoU/FastRCNN/$CONFIG1.py" --gpus 4&&
python tools/test.py "configs/softIoU/MaskAwareIoU/FastRCNN/$CONFIG1.py" "work_dirs/$CONFIG1/epoch12.pth"  --json_out "results/FastRCNN/$CONFIG1/$CONFIG1" --eval bbox &&
python tools/coco_eval.py "results/FastRCNN/$CONFIG1/$CONFIG1.bbox.json" --ann "data/coco/annotations/instances_val2017.json" --LRPEval 1 --LRPtau 0.5 >> "results/FastRCNN/$CONFIG1/$CONFIG1.result"
