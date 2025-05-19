#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : simple_test_result.py
# Author            : WangZi <wangzitju@163.com>
# Date              : 24.11.2020
# Last Modified Date: 24.11.2020
# Last Modified By  : WangZi <wangzitju@163.com>
import json
import cv2
import numpy as np
import os
import os.path as osp
import tqdm
from collections import defaultdict
from utils.utils import Camera
from pymvg import quaternions


COLORS = (
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (79, 129, 189),
    (192, 80, 77),
    (155, 187, 89),
    (128, 100, 162),
    (75, 172, 198),
    (36, 169, 225),
    (91, 74, 66),
    (147, 224, 255),
    (92, 167, 186)
)

img_dir = './data/speed/images/train'
save_dir = './result_test'
gt_file = './data/speed/train.json'


# load world points coordinates
world_pt_file = './data/annos/all_result.json'
with open(world_pt_file, 'r') as f:
    world_pts = json.load(f)
world_pts = [item['pt'] for item in world_pts]


# check output dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

# load ground truth
with open(gt_file, 'r') as f:
    ground_truth = json.load(f)

ground_truth = {
    item['filename']:
    {'quat': item['q_vbs2tango'],
     'tvec': item['r_Vo2To_vbs_true']} for item in ground_truth
}


def find_index(logits):
    """return labels for queries
    Args:
        logits: num_queries x (num_points + 1), the last is bg
    Return:
        index: num_queries,
        score: num_queries,
    """
    # num_queries = logits.shape[0]
    # num_points = logits.shape[1] - 1
    return logits.argmax(1), logits[:, ].max(1)
    # index, scores = np.ones(num_queries) * -1, np.ones(num_queries) * -1

    # # find background
    # bg_indices = logits.argmax(1) == num_points
    # index[bg_indices] = num_points
    # scores[bg_indices] = logits[bg_indices].max(1)

    # # fore ground
    # fg_indices = np.where(logits.argmax(1) < num_points)
    # fg_logits = logits[fg_indices, :num_points-1][0]


with open('./all_ret.json', 'r') as f:
    all_ret = json.load(f)

predict = []

for img_name, ret in tqdm.tqdm(all_ret.items()):
    img = cv2.imread(osp.join(img_dir, img_name))
    assert img is not None, 'can not open {:s}'.format(
        osp.join(img_dir, img_name)
    )
    logits, points = ret['logits'], ret['points']
    logits, points = np.asarray(logits), np.asarray(points)

    labels, scores = find_index(logits)
    fg_indices = labels != logits.shape[1] - 1
    points, labels, scores =\
        points[fg_indices], labels[fg_indices], scores[fg_indices]

    # for pt in points[logits.argmax(1) != 11]:
    P = defaultdict(list)
    for pt, l_, s_ in zip(points, labels, scores):
        l_ = int(l_)
        P[int(l_)].append([pt[0], pt[1], s_])

        x, y = int(pt[0]), int(pt[1])
        # cv2.circle(img, (x, y), 3, COLORS[l_], 3)

    # cv2.imwrite(osp.join(save_dir, img_name), img)

    obj_pts, wld_pts = [], []
    for l_, pt_score in P.items():
        pt_score = np.asarray(pt_score)
        max_idx = pt_score[:, -1].argmax()
        pts = pt_score[max_idx]
        wld_pts.append(world_pts[l_])
        obj_pts.append(pts[:2])

    if len(obj_pts) < 4:
        continue

    obj_pts = np.asarray(obj_pts)[:, np.newaxis, :].astype(np.float32)
    wld_pts = np.asarray(wld_pts)[:, np.newaxis, :].astype(np.float32)

    try:
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            wld_pts, obj_pts, Camera.K, Camera.dist,
            useExtrinsicGuess=False,
            flags=cv2.SOLVEPNP_P3P
        )
        Rmat = cv2.Rodrigues(rvec)[0]
        # quat = quaternions.quaternion_from_matrix(np.linalg.inv(Rmat))
        quat = quaternions.quaternion_from_matrix(Rmat)
    except cv2.error as e:
        print(e)

    predict.append({
        'filename': img_name,
        'quat': quat,
        'tvec': tvec
    })


def speed_score(q_pr, t_pr, q_gt, t_gt):
    q_pr = np.asarray(q_pr)
    t_pr = np.asarray(t_pr)
    q_gt = np.asarray(q_gt)
    t_gt = np.asarray(t_gt)

    s_t = np.linalg.norm(t_pr - t_gt, ord=2) \
        / np.linalg.norm(t_gt, ord=2)

    s_q = 2 * np.abs(np.arccos(q_pr[0]) - np.arccos(q_gt[0]))
    return s_t + s_q


idx = 0
scores = 0
for item in predict:
    filename = item['filename']
    quat_pr, tvec_pr = item['quat'], item['tvec']
    quat_gt = ground_truth[filename]['quat']
    tvec_gt = ground_truth[filename]['tvec']

    scores += speed_score(quat_pr, tvec_pr, quat_gt, tvec_gt)
    idx += 1
print(scores / idx)

