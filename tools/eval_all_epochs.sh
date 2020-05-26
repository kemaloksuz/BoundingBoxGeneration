#!/usr/bin/env bash

cd ~/imgworkspace/mmdetection-test/mmdetection
for ((i = 1 ; i < 13 ; i++)); do
  echo "Evaluating epoch $i"
  python tools/test.py configs/lrp_optimization/initial_experiments/faster_rcnn_r50_fpn_1x_1-IoU_CE.py ./work_dirs/lrp_optimization/faster_rcnn_r50_fpn_1x_1-IoU_lw2_CE/epoch_$i.pth --json_out results/lrp_optimization/faster_rcnn_r50_fpn_1x_1-IoU_lw2_CE/1-IoU_lw2_CE_ep$i --eval bbox
  python tools/coco_eval.py results/lrp_optimization/faster_rcnn_r50_fpn_1x_1-IoU_lw2_CE/1-IoU_lw2_CE_ep$i.bbox.json --ann data/coco/annotations/instances_val2017.json --LRPEval 1 --LRPtau 0.5 >> results/lrp_optimization/faster_rcnn_r50_fpn_1x_1-IoU_lw2_CE/1-IoU_lw2_CE_ep$i.result
  sleep 30s
done
