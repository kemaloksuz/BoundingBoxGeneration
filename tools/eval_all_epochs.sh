#!/usr/bin/env bash
EXP_NAME="fast_rcnn_r50_fpn_1x_IoU_CE_minicoco";
for ((i=21; i<=36; i++)); do
  echo "Evaluating epoch $i"
  python tools/test.py configs/lrp_optimization/fast_rcnn/$EXP_NAME.py work_dirs/lrp_optimization/$EXP_NAME/epoch_$i.pth --json_out results/lrp_optimization/$EXP_NAME/epoch_$i --eval bbox &&
  python tools/coco_eval.py results/lrp_optimization/$EXP_NAME/epoch_$i.bbox.json --ann data/coco/annotations/instances_val2017.json --LRPEval 1 --LRPtau 0.5 >> results/lrp_optimization/$EXP_NAME/epoch_$i.result
done

