#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : speed_eval.py
# Author            : WangZi <wangzitju@163.com>
# Date              : 23.11.2020
# Last Modified Date: 23.11.2020
# Last Modified By  : WangZi <wangzitju@163.com>
"""
SPEED 数据集的eval
包括位姿求解方法
"""
import json
import numpy as np
from utils.utils import Camera
from collections import defaultdict
import cv2
import mathutils
from scipy.spatial.distance import cdist


def build_solver(args):
    return SimplePoseSolver(args)


world_pt_path = './data/annos/all_result.json'


class PoseSolver(object):
    """
    Solve pose according to the output from DETR
    """

    def __init__(self):
        with open(world_pt_path, 'r') as f:
            world_pt = json.load(f)
        W_Pt = []
        for item in world_pt:
            W_Pt.append(item['pt'])
        self.W_Pt = np.asarray(W_Pt)


class Multi_Mean_PoseSolver(PoseSolver):
    def __init__(self, args):
        super(Multi_Mean_PoseSolver, self).__init__()
        self.reprojectionError = args.repro

    def find_index(self, logits):
        """return labels for queries
        Args:
            logits: num_queries x (num_points + 1), the last is bg
        Return:
            index: num_queries,
            score: num_queries,
        """
        # num_queries = logits.shape[0]
        # num_points = logits.shape[1] - 1
        return logits.argmax(1), logits.max(1)

    def mean_and_filter(self, obj_pts):
        """每个控制点的所有预测像点求平均，并以sigma为准则剔除粗大误差
        """
        result = {}
        for label, points in obj_pts.items():
            num_points = len(points)
            points = np.vstack(points)
            points_mean = np.mean(points, axis=0, keepdims=True)
            if num_points < 3:
                result[label] = points_mean.flatten()
            else:
                distances = cdist(points, points_mean).flatten()
                std_dist = np.std(distances)
                inlier_index = distances < std_dist * 3  # 3 sigma
                if not np.all(inlier_index):
                    print('{:d} points: all larger that 3 sigma'.format(label))
                result[label] = np.mean(points[inlier_index], axis=0).flatten()
        return result

    def __call__(self, multi_points, multi_logits):
        assert isinstance(multi_points, list)
        assert isinstance(multi_logits, list)
        assert len(multi_points) == len(multi_logits)

        obj_pts_original = defaultdict(list)
        for points, logits in zip(multi_points, multi_logits):
            labels, scores = self.find_index(logits)
            fg_indices = labels != logits.shape[1] - 1
            points, labels, scores =\
                points[fg_indices], labels[fg_indices], scores[fg_indices]

            for pt_, l_ in zip(points, labels):
                obj_pts_original[l_].append(pt_)

        obj_pts_mean = self.mean_and_filter(obj_pts_original)

        obj_pts, wld_pts = [], []
        for l_, pts in obj_pts_mean.items():
            wld_pts.append(self.W_Pt[l_])
            obj_pts.append(pts[:2])

        try:
            obj_pts = np.asarray(obj_pts)[:, np.newaxis, :].astype(np.float32)
            wld_pts = np.asarray(wld_pts)[:, np.newaxis, :].astype(np.float32)
        except IndexError as e:
            raise e

        try:
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                wld_pts, obj_pts, Camera.K, Camera.dist,
                useExtrinsicGuess=False,
                flags=cv2.SOLVEPNP_P3P,
                reprojectionError=self.reprojectionError
            )

            if np.any(np.isnan(rvec)):
                print(rvec)

            if inliers is not None:
                retval, rvec, tvec, _ = cv2.solvePnPGeneric(
                    wld_pts[inliers.flatten()],
                    obj_pts[inliers.flatten()],
                    Camera.K, Camera.dist,
                    useExtrinsicGuess=True,
                    rvec=rvec,
                    tvec=tvec,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                assert len(rvec) == 1, 'find more than one solution in ransac'
                rvec, tvec = rvec[0], tvec[0]

            Rmat = cv2.Rodrigues(rvec)[0]
            quat = mathutils.Matrix(Rmat).to_quaternion()
            # quat = quaternions.quaternion_from_matrix(np.linalg.inv(Rmat))
            quat = np.asarray(quat)
        except cv2.error as e:
            raise e

        if np.any(np.isnan(quat)):
            print(rvec, quat)

        return quat.flatten(), tvec.flatten()


class SimplePoseSolver(PoseSolver):
    """
    Solve pose according to the output from DETR
    """

    def __init__(self, args):
        super(SimplePoseSolver, self).__init__()
        self.reprojectionError = args.repro

    def find_index(self, logits):
        """return labels for queries
        Args:
            logits: num_queries x (num_points + 1), the last is bg
        Return:
            index: num_queries,
            score: num_queries,
        """
        # num_queries = logits.shape[0]
        # num_points = logits.shape[1] - 1
        return logits.argmax(1), logits.max(1)

    def __call__(self, points, logits):
        """ get quat and tra from the output of DETR
        Args:
            points: ndarray (num_queries x 2),
                    according to coordinates in original image
            logits: ndarray (num_queries x 12),
                    points classification results
            return: result format in speed
                - quat: quaterion
                - tra: translation
        """
        assert isinstance(points, (np.ndarray, list))
        assert isinstance(logits, (np.ndarray, list))
        if isinstance(points, list):
            points = np.asarray(points)
        if isinstance(logits, list):
            logits = np.asarray(logits)

        assert points.shape[0] == logits.shape[0], '[Solver]: num_queries!'

        labels, scores = self.find_index(logits)
        fg_indices = labels != logits.shape[1] - 1
        points, labels, scores =\
            points[fg_indices], labels[fg_indices], scores[fg_indices]

        P = defaultdict(list)
        for pt, l_, s_ in zip(points, labels, scores):
            l_ = int(l_)
            P[int(l_)].append([pt[0], pt[1], s_])

        obj_pts, wld_pts = [], []
        for l_, pt_score in P.items():
            pt_score = np.asarray(pt_score)
            max_idx = pt_score[:, -1].argmax()
            pts = pt_score[max_idx]
            wld_pts.append(self.W_Pt[l_])
            obj_pts.append(pts[:2])

        try:
            obj_pts = np.asarray(obj_pts)[:, np.newaxis, :].astype(np.float32)
            wld_pts = np.asarray(wld_pts)[:, np.newaxis, :].astype(np.float32)
        except IndexError as e:
            raise e

        try:
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                wld_pts, obj_pts, Camera.K, Camera.dist,
                useExtrinsicGuess=False,
                flags=cv2.SOLVEPNP_P3P,
                reprojectionError=self.reprojectionError
            )

            if np.any(np.isnan(rvec)):
                print(rvec)

            if inliers is not None:
                retval, rvec, tvec, _ = cv2.solvePnPGeneric(
                    wld_pts[inliers.flatten()],
                    obj_pts[inliers.flatten()],
                    Camera.K, Camera.dist,
                    useExtrinsicGuess=True,
                    rvec=rvec,
                    tvec=tvec,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                assert len(rvec) == 1, 'find more than one solution in ransac'
                rvec, tvec = rvec[0], tvec[0]

            Rmat = cv2.Rodrigues(rvec)[0]
            quat = mathutils.Matrix(Rmat).to_quaternion()
            # quat = quaternions.quaternion_from_matrix(np.linalg.inv(Rmat))
            quat = np.asarray(quat)
        except cv2.error as e:
            raise e

        if np.any(np.isnan(quat)):
            print(rvec, quat)

        return quat.flatten(), tvec.flatten()


def speed_score(q_pr, t_pr, q_gt, t_gt):
    q_pr = np.asarray(q_pr).flatten()
    t_pr = np.asarray(t_pr).flatten()
    q_gt = np.asarray(q_gt).flatten()
    t_gt = np.asarray(t_gt).flatten()

    assert q_pr.shape[0] == q_gt.shape[0] == 4
    assert t_pr.shape[0] == t_gt.shape[0] == 3
    if q_pr[0] < 0:
        q_pr = q_pr * -1
    if q_gt[0] < 0:
        q_gt = q_gt * -1

    s_t = np.linalg.norm(t_pr - t_gt, ord=2) \
        / np.linalg.norm(t_gt, ord=2)

    s_q = 2 * np.arccos(min(np.abs(np.dot(q_pr, q_gt)), 1))
    return s_t, s_q
