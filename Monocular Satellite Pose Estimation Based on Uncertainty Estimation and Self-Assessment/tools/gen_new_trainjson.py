import os.path as osp

DATA_ROOT = "/media/willer/ST1/datasets/speed/speed/"
ann_file = "wz_train.json"
output_file = "wz_train_del_0_1.json"
import json


def load_del_0_1(ann_file, output_file):
    """read detection result from mmdetection
    Input:
        {
            "img000424real.jpg": [[x1, y1, x2, y2, score]],
        }
    Return:
        [
            {'filename': 'img000231.jpg', 'bbox_xxyy': [x1, y1, x2, y2],
            .
            .
        ]
    """
    with open(osp.join(DATA_ROOT, "annos", ann_file), "r") as f1:
        anns = json.load(f1)
    for i, image in enumerate(anns):
        anns[i]["landmarks"] = image["landmarks"][2:]
    with open(osp.join(DATA_ROOT, "annos", output_file), "w") as f2:
        json.dump(anns, f2)


load_del_0_1(ann_file, output_file)
# out = []
# for filename, bbox_score in anns.items():
#     out.append({"filename": filename, "bbox_xxyy": bbox_score[0][:4]})
