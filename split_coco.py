from pycocotools.coco import COCO
import argparse
import numpy as np

import random
import os
import json
import pdb

coco_output = {
        "categories": [],
        "images": [],
        "annotations": []
        }

nofimages = 1000

annotation_path = "./data/coco/annotations"
annotation_name = "instances_train2017"
annotations_root = os.path.join(annotation_path, annotation_name+".json")
coco = COCO(annotations_root)
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
coco_output["categories"] = cats
catIds = coco.getCatIds(catNms=nms)
image_ids = coco.getImgIds()
pdb.set_trace()
image_ids = random.sample(image_ids,nofimages)

for image_id in image_ids:
    image_info = coco.loadImgs(image_id)[0]
    coco_output["images"].append(image_info)
    annotation_ids = coco.getAnnIds(imgIds=image_id)
    for annotation_id in annotation_ids:
        annotation_info = coco.loadAnns(annotation_id)[0]
        coco_output["annotations"].append(annotation_info)

save_name = os.path.join(annotation_path, annotation_name+".json")
with open(save_name, 'w') as output_json_file:
    json.dump(coco_output, output_json_file)

