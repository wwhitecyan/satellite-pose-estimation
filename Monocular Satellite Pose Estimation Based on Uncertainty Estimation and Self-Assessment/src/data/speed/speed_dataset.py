#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : speed.py
# Author            : WangZi <wangzitju@163.com>
# Date              : 21.11.2020
# Last Modified Date: 03.12.2020
# Last Modified By  : WangZi <wangzitju@163.com>
import json
import os.path as osp
import numpy as np
from PIL import Image
from scipy import stats
import cv2

import torch
from torch.utils.data import Dataset
import albumentations as A
import torchvision.transforms.functional as F
from utils.speed_eval import speed_score
from src.core import register

from utils.speed_eval import build_solver
from utils.speed_eval import build_sigma_solver
from utils.speed_eval_ceres import build_epnp_sigma_solver

# from utils.speed_eval_ceres import build_solver

__all__ = ["SpeedSubmission", "SpeedTrain", "SpeedEval", "speed_collate_fn"]
DATA_ROOT = "/media/willer/ST1/datasets/speed/speed/"


@register
def speed_collate_fn(batch):
    batch = list(zip(*batch))
    # batch[0] = nested_tensor_from_tensor_list(batch[0])
    batch[0] = torch.stack(batch[0], 0)
    return tuple(batch)


class Normalize(object):
    def __init__(self, mean, std, size=512):
        self.mean = mean
        self.std = std
        self.size = size

    def __call__(self, image, landmarks=None):
        """
        image: tensor CxHxW
        landmarks: tensors: 11x2
        """
        image = F.normalize(image, mean=self.mean, std=self.std)
        if landmarks is None:
            return image, None

        landmarks = landmarks / self.size
        return image, landmarks


class SpeedSubmission(Dataset):
    def __init__(self, ann_file, img_dir, resize=512):
        self.ann_file, self.img_dir = ann_file, img_dir
        self.resize = resize
        self.transforms = A.Compose(
            [A.Resize(resize, resize, cv2.INTER_CUBIC)],
        )
        self.normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], resize)

        self.data_infos = self.load_anns()

    def load_anns(self):
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
        with open(osp.join(DATA_ROOT, "annos", self.ann_file), "r") as f:
            anns = json.load(f)

        out = []
        for filename, bbox_score in anns.items():
            out.append({"filename": filename, "bbox_xxyy": bbox_score[0][:4]})

        return out

    """
    2020.12.14换一种crop的策略。之前是将bbox_clip按照图像边缘进行压缩，
    这样会改变输入图像的ratio。
    新的策略，是按照图像边缘进行pad，使裁剪后的图像保持输入图像的ratio.
    1. 首先，generate_clip_bbox 函数仅输出裁剪的bbox，不考虑图像边界
    2. 然后，生成一个大小域bbox_clip相同的tensor，将图像拷贝进去
    """

    def generate_clip_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        bbox_width, bbox_height = x2 - x1, y2 - y1
        scale = max(bbox_width, bbox_height) * 1.2

        x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
        half_scale = scale / 2

        # x1, x2 = x_center - half_scale, x_center + half_scale
        # y1, y2 = y_center - half_scale, y_center + half_scale
        # bbox_clip = np.asarray([x1, y1, x2, y2])
        # bbox_clip = np.floor(bbox_clip)
        x1, y1 = int(x_center - half_scale), int(y_center - half_scale)
        scale = int(scale)
        bbox_clip = np.asarray([x1, y1, x1 + scale, y1 + scale])

        return bbox_clip

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        img_path = osp.join(DATA_ROOT, self.img_dir, data_info["filename"])
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        bbox_xxyy = data_info["bbox_xxyy"]

        # 1. 确定裁剪图像
        bbox_clip = self.generate_clip_bbox(bbox_xxyy)
        clip_size = int(bbox_clip[2] - bbox_clip[0])
        img = np.asarray(img)

        # 2. 生成canvas
        cropImg = np.zeros((clip_size, clip_size, 3), dtype=img.dtype)
        x1 = max(0, bbox_clip[0])
        crop_x1 = x1 - bbox_clip[0]
        y1 = max(0, bbox_clip[1])
        crop_y1 = y1 - bbox_clip[1]
        x2 = min(width, bbox_clip[2])
        # crop_x2 = clip_size - (bbox_clip[2] - x2 - 1)
        y2 = min(height, bbox_clip[3])
        # crop_y2 = clip_size - (bbox_clip[3] - y2 - 1)
        crop_x1 = int(crop_x1)
        crop_y1 = int(crop_y1)
        # crop_x1, crop_x2, crop_y1, crop_y2 = [
        #     int(item) for item in (crop_x1, crop_x2, crop_y1, crop_y2)]

        x1, x2, y1, y2 = [int(item) for item in (x1, x2, y1, y2)]
        # cropImg[crop_y1:crop_y2, crop_x1:crop_x2] = img[y1:y2, x1:x2]
        cropImg[crop_y1 : crop_y1 + y2 - y1, crop_x1 : crop_x1 + x2 - x1] = img[
            y1:y2, x1:x2
        ]

        if self.transforms:
            transformed = self.transforms(image=cropImg)
            cropImg = transformed["image"]
        # cropImg = torch.from_numpy(
        #     cropImg).to(torch.float).permute(2, 0, 1) / 255
        cropImg = F.to_tensor(cropImg)
        cropImg, _ = self.normalize(cropImg)

        target = {
            "clip_bbox": torch.as_tensor(bbox_clip),
            "filename": data_info["filename"],
        }

        return cropImg, target


def img_trunc(img, ratio=0.2, p=0.1):
    """
    truncate img by the ratio
    """
    if np.random.rand() > p:
        return img
    height, width = img.shape[:2]
    trunc_height, trunc_width = height * ratio, width * ratio
    trunc_height = np.random.randint(0, int(trunc_height))
    trunc_width = np.random.randint(0, int(trunc_width))
    tmp = np.random.rand()

    # 1. truncate height
    if tmp < 0.25:
        img[:trunc_height, :] = 0
    if tmp > 0.75:
        img[-trunc_height:, :] = 0
    # 2. truncate width
    if tmp < 0.25:
        img[:, :trunc_width] = 0
    if tmp > 0.75:
        img[:, -trunc_width:] = 0
    return img


@register
class SpeedTrain(Dataset):
    def __init__(self, ann_file, index_file, img_dir, resize=512, train=True):
        super(SpeedTrain, self).__init__()
        self.ann_file, self.img_dir = ann_file, img_dir
        self.index_file = index_file
        self.transforms = make_transforms(train, resize)

        self.train, self.resize = train, resize

        self.data_infos = self.load_anns()
        self.normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], resize)

    def load_anns(self):
        with open(osp.join(DATA_ROOT, "annos", self.ann_file), "r") as f:
            anns = json.load(f)

        indexs = np.loadtxt(osp.join(DATA_ROOT, "annos", self.index_file), dtype=int)
        import ipdb

        ipdb.set_trace()

        return [anns[item] for item in indexs.tolist()]

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        img_path = osp.join(DATA_ROOT, self.img_dir, data_info["filename"])
        img = Image.open(img_path).convert("RGB")

        # qua, tra = data_info['q_vbs2tango'], data_info['r_Vo2To_vbs_true']
        landmarks = np.asarray(data_info["landmarks"])  # 11 x 2
        bbox_xxyy = data_info["bbox_xxyy"]

        if self.train:
            bbox_clip = self.generate_clip_bbox_train(bbox_xxyy, img.size)
        else:
            bbox_clip = self.generate_clip_bbox_val(bbox_xxyy, img.size)
        landmarks[:, :2] -= bbox_clip[:2]
        cropImg = np.array(img.crop(bbox_clip))

        if self.transforms:
            transformed = self.transforms(image=cropImg, keypoints=landmarks)
            cropImg, landmarks = transformed["image"], transformed["keypoints"]

        # cropImg = torch.from_numpy(
        #     cropImg).to(torch.float).permute(2, 0, 1) / 255
        cropImg = img_trunc(cropImg, p=0.2)
        cropImg = F.to_tensor(cropImg)
        landmarks = torch.as_tensor(landmarks)
        cropImg, landmarks = self.normalize(cropImg, landmarks)

        target = {
            "landmarks": landmarks.to(torch.float32),
            "clip_bbox": torch.as_tensor(bbox_clip),
            "labels": torch.arange(0, 11, dtype=torch.int64),
            "filename": data_info["filename"],
        }

        return cropImg, target

    def generate_clip_bbox_val(self, bbox, image_size):
        x1, y1, x2, y2 = bbox
        bbox_width, bbox_height = x2 - x1, y2 - y1
        scale = max(bbox_width, bbox_height) * 1.2

        x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
        half_scale = scale / 2

        x1, x2 = x_center - half_scale, x_center + half_scale
        y1, y2 = y_center - half_scale, y_center + half_scale
        bbox_clip = np.asarray([x1, y1, x2, y2])

        bbox_clip[0::2] = bbox_clip[0::2].clip(min=0, max=image_size[0])
        bbox_clip[1::2] = bbox_clip[1::2].clip(min=0, max=image_size[1])
        return bbox_clip

    def generate_clip_bbox_train(self, bbox, image_size):
        """
        generate clip bboxes
        bbox: (4,), x1y1x2y2
        image_size: (width, height)
        return: np.ndarray nx4
        """
        alpha = 0.2
        beta = 0.2
        center = np.asarray([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])

        bbox_width, bbox_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        scale = max(bbox_width, bbox_height) * 1.2

        # truncnorm
        # 1 x_center
        x_center = int(truncnorm(center[0], bbox_width * alpha))
        y_center = int(truncnorm(center[1], bbox_height * alpha))
        scale = int(truncnorm(scale, scale * beta))
        half_scale = np.ceil(scale / 2.0)

        x1, x2 = x_center - half_scale, x_center + half_scale
        y1, y2 = y_center - half_scale, y_center + half_scale

        bbox_clip = np.asarray([x1, y1, x2, y2])

        bbox_clip[0::2] = bbox_clip[0::2].clip(min=0, max=image_size[0])
        bbox_clip[1::2] = bbox_clip[1::2].clip(min=0, max=image_size[1])
        return bbox_clip

    def __len__(self):
        return len(self.data_infos)


def truncnorm(loc, clip):
    mu, sigma = loc, clip / 3
    lower, upper = mu - 3 * sigma, mu + 3 * sigma  # 截断在[μ-3σ, μ+3σ]
    X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    x = X.rvs(1)
    return x


def make_transforms(train=True, img_size=512):
    if train:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Affine(p=0.1, mode=cv2.BORDER_CONSTANT),
                A.Rotate(
                    p=0.4,
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                A.Resize(img_size, img_size, cv2.INTER_CUBIC),
                A.Blur(p=0.3, blur_limit=3),
                A.OneOf(
                    [
                        A.MotionBlur(p=0.2),
                        A.MedianBlur(blur_limit=3, p=0.1),
                        A.Blur(blur_limit=3, p=0.1),
                    ],
                    p=0.2,
                ),
                # A.IAAAdditiveGaussianNoise(p=0.2),
                A.GaussNoise(p=0.2),
                A.JpegCompression(20, 99, p=0.1),
                A.ISONoise(p=0.2),
                A.Cutout(num_holes=10, max_h_size=40, max_w_size=40, p=0.1),
            ],
            keypoint_params=A.KeypointParams("xy", remove_invisible=False),
        )
    else:
        return A.Compose(
            [A.Resize(img_size, img_size, cv2.INTER_CUBIC)],
            keypoint_params=A.KeypointParams("xy", remove_invisible=False),
        )


@register
class SpeedEval(object):
    def __init__(self, ann_file, index_file):
        # self.solver = build_solver()
        self.solver = build_sigma_solver()
        # self.solver = build_epnp_sigma_solver()
        self.ann_file = ann_file
        self.index_file = index_file
        self.data_infos = self.load_anns()

        with open(osp.join(DATA_ROOT, "annos", ann_file), "r") as f:
            ground_truth = json.load(f)
        # import ipdb

        # ipdb.set_trace()

        self.ground_truth = {
            item["filename"]: {
                "quat": item["q_vbs2tango"],
                "tvec": item["r_Vo2To_vbs_true"],
                "area": np.sqrt(
                    (item["bbox_xxyy"][2] - item["bbox_xxyy"][0]) * item["bbox_xxyy"][3]
                    - item["bbox_xxyy"][1]
                ),
            }
            for item in ground_truth
        }
        self.log = {}

    def load_anns(self):
        with open(osp.join(DATA_ROOT, "annos", self.ann_file), "r") as f:
            anns = json.load(f)

        indexs = np.loadtxt(osp.join(DATA_ROOT, "annos", self.index_file), dtype=int)

        return [anns[item] for item in indexs.tolist()]

    def __len__(self):
        return len(self.data_infos)

    def update(self, predictions, aux_outputs):
        # def update(self, predictions):
        for filename, ret in predictions.items():
            try:
                # sigma = np.ones_like(ret["points"])
                sigma = ret["sigmas"]
                area = self.ground_truth[filename]["area"]

                # quat_pr, tvec_pr = self.solver(ret["points"], ret["logits"])
                quat_pr, tvec_pr = self.solver(ret["points"], ret["logits"], sigma)
                # quat_pr, tvec_pr = self.solver(
                #     ret["points"], ret["logits"], area, sigma
                # )
            # except cv2.error or IndexError as e:
            except IndexError as e:
                quat_pr, tvec_pr = np.zeros(4), np.zeros(3)
                # print('{:s}: '.format(filename), end='')
                # print(e)
            except cv2.error as e:
                quat_pr, tvec_pr = np.zeros(4), np.zeros(3)
                # print('{:s}: '.format(filename), end='')
                # print(e)

            quat_gt = self.ground_truth[filename]["quat"]
            tvec_gt = self.ground_truth[filename]["tvec"]

            score_tvec, score_quat = speed_score(quat_pr, tvec_pr, quat_gt, tvec_gt)
            aux_0 = aux_outputs[0][filename]
            aux_1 = aux_outputs[1][filename]
            aux_2 = aux_outputs[2][filename]

            # print(
            #     "\n start aux_0:", aux_0, "\n \t aux_1:", aux_1, "\n \t aux_2:", aux_2
            # )
            self.log[filename] = {
                "points": np.around(ret["points"], decimals=2).tolist(),
                "aux_points_0": np.around(aux_0, decimals=2).tolist(),
                "aux_points_1": np.around(aux_1, decimals=2).tolist(),
                "aux_points_2": np.around(aux_2, decimals=2).tolist(),
                "logits": np.around(ret["logits"], decimals=6).tolist(),
                "quat_gt": quat_gt,
                "tvec_gt": tvec_gt,
                "quat_pr": np.around(quat_pr, decimals=6).tolist(),
                "tvec_pr": np.around(tvec_pr, decimals=6).tolist(),
                "score_tvec": np.around(score_tvec, decimals=8).item(),
                "score_quat": np.around(score_quat, decimals=8).item(),
                "score": np.around(score_quat + score_tvec, decimals=8).item(),
                "sigma": np.around(sigma, decimals=8).tolist(),
            }

    def summarize(self):
        scores = np.asarray([item["score"] for filename, item in self.log.items()])
        tvec_score = np.asarray(
            [item["score_tvec"] for filename, item in self.log.items()]
        )
        quat_score = np.asarray(
            [item["score_quat"] for filename, item in self.log.items()]
        )

        tvec_abs = np.stack(
            [
                np.abs(np.asarray(item["tvec_pr"]) - np.asarray(item["tvec_gt"]))
                for _, item in self.log.items()
            ]
        )

        scores = np.mean(scores).item()
        tvec_score = np.mean(tvec_score).item()
        quat_score = np.mean(quat_score).item()

        # score (mean)
        self.stats = (
            "tvec score: {:.6f}, "
            "quat score: {:.6f}, final score: {:.6f}; ".format(
                tvec_score, quat_score, scores
            )
        )

        # median error
        self.stats = (
            self.stats + "median tvec: {:.6f}, "
            "median quat: {:.6f}; ".format(
                np.median(tvec_score).item(), np.median(quat_score).item()
            )
        )

        tvec_abs_mean = np.mean(tvec_abs, 0).tolist()
        tvec_abs_median = np.median(tvec_abs, 0).tolist()
        # tvec abs
        self.stats = (
            self.stats + "mean tvec abs: "
            "[{:.6f}, {:.6f}, {:.6f}], median tvec abs:"
            "[{:.6f}, {:.6f}, {:.6f}]".format(*(tvec_abs_mean + tvec_abs_median))
        )
