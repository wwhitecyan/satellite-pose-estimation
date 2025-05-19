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
import itertools
import json
from collections import defaultdict
from functools import partial

import cv2
import mathutils
import numpy as np
import PyCeres
from scipy.spatial.distance import cdist

from .utils import Camera


def build_epnp_sigma_solver(input_size=256):
    return EPnPCeresSolver(input_size)


# def build_solver(args):
#     return SimplePoseSolver(args)


world_pt_path = "/media/willer/ST1/datasets/speed/annos/all_result.json"
with open(world_pt_path, "r") as f:
    world_pt = json.load(f)
W_Pt = []
for item in world_pt:
    W_Pt.append(item["pt"])
W_Pt = np.asarray(W_Pt)


class EPnPCeresSolver(object):
    def __init__(self, input_size):
        with open(world_pt_path, "r") as f:
            world_pt = json.load(f)
        W_Pt = []
        for item in world_pt:
            W_Pt.append(item["pt"])
        self.W_Pt = np.asarray(W_Pt)
        self.input_size = input_size

    def get_repro_th(self, area):
        """根据2D检测的bounding box的区域面积，设置重投影残差阈值确定外点"""
        repro = int(area / self.input_size * 10)
        repro = min(max(repro, 1.5), 20)
        self.reprojectionError = repro
        # self.reprojectionError = 25

    def find_index(self, logits):
        """return labels for queries
        Args:
            logits: num_queries x (num_points + 1), the last is bg
        Return:
            index: num_queries,
            score: num_queries,
        """
        return logits.argmax(1), logits.max(1)

    def __call__(self, points, logits, area, sigma):
        """get quat and tra from the output of DETR
        Args:
            points: ndarray (num_queries x 2),
                    according to coordinates in original image
            logits: ndarray (num_queries x 12),
                    points classification results
            area: int, bounding box area
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

        assert points.shape[0] == logits.shape[0], "[Solver]: num_queries!"
        self.get_repro_th(area)

        labels, scores = self.find_index(logits)
        fg_indices = labels != logits.shape[1] - 1
        points, labels, scores, sigma = (
            points[fg_indices],
            labels[fg_indices],
            scores[fg_indices],
            sigma[fg_indices],
        )

        P = defaultdict(list)
        for pt, l_, s_, sig in zip(points, labels, scores, sigma):
            l_ = int(l_)
            P[int(l_)].append([pt[0], pt[1], s_, sig])

        obj_pts, wld_pts, sigma_select = [], [], []
        for l_, pt_score in P.items():
            pt_score = np.asarray(pt_score)
            max_idx = pt_score[:, -2].argmax()
            # for x_, y_, s_, sig in pt_score:
            #     if s_ > 0.5 and sig.mean() < 5:
            #         wld_pts.append(self.W_Pt[l_])
            #         obj_pts.append([x_, y_])
            #         sigma_select.append(sig)
            pts = pt_score[max_idx]
            wld_pts.append(self.W_Pt[l_])
            obj_pts.append(pts[:2])
            sigma_select.append(pts[-1])
        sigma_select = np.asarray(sigma_select)

        try:
            obj_pts = np.asarray(obj_pts)[:, np.newaxis, :].astype(np.float32)
            wld_pts = np.asarray(wld_pts)[:, np.newaxis, :].astype(np.float32)
        except IndexError as e:
            raise e
        # obj_pts.shape num_queries-background,1,2
        # wld_pts.shape num_queries-background,1,3
        (
            rvec_init,
            tvec_init,
            inlier_index,
            rep_err_before_lm,
        ) = self.epnp_init(obj_pts, wld_pts)

        inlier_obj = obj_pts[inlier_index]
        inlier_wld = wld_pts[inlier_index]
        inlier_sigma = sigma_select[inlier_index]

        rvec, tvec = ceres_pnp(
            inlier_wld, inlier_obj, inlier_sigma, rvec_init, tvec_init
        )
        obj_rep, _ = cv2.projectPoints(wld_pts, rvec, tvec, Camera.K, Camera.dist)
        rep_err = np.linalg.norm(obj_rep - obj_pts, axis=-1).sum()
        if rep_err > rep_err_before_lm:
            rvec, tvec = rvec_init, tvec_init
            rep_err = rep_err_before_lm

        Rmat = cv2.Rodrigues(rvec)[0]
        quat = mathutils.Matrix(Rmat).to_quaternion()
        quat = np.asarray(quat)
        return quat.flatten(), tvec.flatten()

    def epnp_init(self, obj_pts, wld_pts):
        # EPnP
        retval, r_, t_, repro_error = cv2.solvePnPGeneric(
            wld_pts,
            obj_pts,
            Camera.K,
            Camera.dist,
            flags=cv2.SOLVEPNP_EPNP
            # wld_pts, obj_pts, Camera.K, Camera.dist, flags=cv2.SOLVEPNP_P3P
        )
        obj_rep, _ = cv2.projectPoints(wld_pts, r_[0], t_[0], Camera.K, Camera.dist)
        rep_err_before_lm = np.linalg.norm(obj_rep - obj_pts, axis=-1)
        rep_err_before_lm = rep_err_before_lm.squeeze()
        inlier_index = np.where(rep_err_before_lm < self.reprojectionError)[0]
        rep_err_before_lm = np.sum(rep_err_before_lm).item()

        return r_[0], t_[0], inlier_index, rep_err_before_lm


def ceres_pnp(wld_pts, obj_pts, sigma, rvec, tvec):
    """
    obj_pts: Nx1x2
    wld_pts: Nx1x3
    """
    obj_pts = cv2.undistortPoints(obj_pts, Camera.K, Camera.dist)
    obj_pts, wld_pts = np.squeeze(obj_pts), np.squeeze(wld_pts)
    num_points = len(obj_pts)
    camera = np.concatenate((rvec.flatten(), tvec.flatten()))

    # print(sigma)
    # sigma[sigma < 1] = 1
    # import ipdb
    # ipdb.set_trace()

    # sigma = np.sqrt(sigma)
    # sigma_lablace = np.exp(-sigma)
    # sigma = sigma_lablace / np.sum(sigma_lablace, axis=0)

    sigma = np.sqrt(sigma)
    sigma_w1 = 1 / (sigma + 1e-6)
    sigma_sum = np.sum(sigma_w1, axis=0)
    sigma = sigma_w1 / sigma_sum
    # sigma_w3 = sigma_w1.copy()
    # sigma_w3[:, 0] = sigma_w3[:, 0] / sigma_sum[0]
    # sigma_w3[:, 1] = sigma_w3[:, 1] / sigma_sum[1]

    problem = PyCeres.Problem()
    for idx in range(num_points):
        loss = PyCeres.HuberLoss(0.001)

        cost_function = PyCeres.CreatePnPCostFunction(
            obj_pts[idx, 0],
            sigma[idx, 0],
            # sigma[idx, 0],
            obj_pts[idx, 1],
            sigma[idx, 1],
            # sigma[idx, 1],
            wld_pts[idx, 0],
            wld_pts[idx, 1],
            wld_pts[idx, 2],
        )
        problem.AddResidualBlock(cost_function, loss, camera)

    # sigma = np.sqrt(sigma)
    # # sigma = 1 / np.exp(-sigma)
    # problem = PyCeres.Problem()
    # for idx in range(num_points):
    #     loss = PyCeres.HuberLoss(0.001)

    #     cost_function = PyCeres.CreatePnPCostFunction(
    #         obj_pts[idx, 0],
    #         1 / (sigma[idx, 0] + 1e-6),
    #         # sigma[idx, 0],
    #         obj_pts[idx, 1],
    #         1 / (sigma[idx, 1] + 1e-6),
    #         # sigma[idx, 1],
    #         wld_pts[idx, 0],
    #         wld_pts[idx, 1],
    #         wld_pts[idx, 2],
    #     )
    #     problem.AddResidualBlock(cost_function, loss, camera)
    options = PyCeres.SolverOptions()
    options.linear_solver_type = PyCeres.LinearSolverType.DENSE_QR
    options.max_num_iterations = 20
    # options.minimizer_progress_to_stdout = True

    summary = PyCeres.Summary()
    PyCeres.Solve(options, problem, summary)
    # print(summary.FullReport())
    # print(camera)
    return np.asarray(camera[:3]), np.asarray(camera[3:])


class PoseSolver(object):
    """
    Solve pose according to the output from DETR
    """

    def __init__(self, args):
        with open(world_pt_path, "r") as f:
            world_pt = json.load(f)
        W_Pt = []
        for item in world_pt:
            W_Pt.append(item["pt"])
        self.W_Pt = np.asarray(W_Pt)
        self.input_size = args.input_size

        self.topk = args.topk
        self.reprojectionError = args.repro

        if args.pnp_method == "ransac":
            self.solve_pnp = self.simple_ransac_pnp
        elif args.pnp_method == "exhausive":
            self.solve_pnp = self.exhausive_pnp
        else:
            raise ValueError("error pnp type")

    def get_repro_th(self, area):
        """根据2D检测的bounding box的区域面积，设置重投影残差阈值确定外点"""
        repro = int(area / self.input_size * 10)
        repro = min(max(repro, 1.5), 20)
        self.reprojectionError = repro

    def find_index(self, logits):
        """return labels for queries
        Args:
            logits: num_queries x (num_points + 1), the last is bg
        Return:
            index: num_queries,
            score: num_queries,
        """
        return logits.argmax(1), logits.max(1)

    def simple_ransac_pnp(self, obj_pts, wld_pts):
        try:
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                wld_pts,
                obj_pts,
                Camera.K,
                Camera.dist,
                useExtrinsicGuess=False,
                flags=cv2.SOLVEPNP_P3P,
                reprojectionError=self.reprojectionError,
            )

            if np.any(np.isnan(rvec)):
                print(rvec)

            if inliers is not None:
                retval, rvec, tvec, _ = cv2.solvePnPGeneric(
                    wld_pts[inliers.flatten()],
                    obj_pts[inliers.flatten()],
                    Camera.K,
                    Camera.dist,
                    useExtrinsicGuess=True,
                    rvec=rvec,
                    tvec=tvec,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                assert len(rvec) == 1, "find more than one solution in ransac"
                rvec, tvec = rvec[0], tvec[0]

            Rmat = cv2.Rodrigues(rvec)[0]
            quat = mathutils.Matrix(Rmat).to_quaternion()
            quat = np.asarray(quat)
        except cv2.error as e:
            raise e

        if np.any(np.isnan(quat)):
            print(rvec, quat)

        return quat.flatten(), tvec.flatten()

    def exhausive_pnp(self, obj_pts, wld_pts):
        all_e, all_q, all_t = [], [], []

        # obj_chosed_all, wld_chosed_all = [], []
        # for indice in itertools.combinations(range(len(obj_pts)), 4):
        #     obj_chosed_all.append(obj_pts[indice])
        #     wld_chosed_all.append(wld_pts[indice])
        # pool = ProcessPoolExecutor(max_workers=8)
        # cv_pnp = partial(
        #     cv2.solvePnPGeneric,
        #     cameraMatrix=Camera.K,
        #     distCoeffs=Camera.dist,
        #     flags=cv2.SOLVEPNP_EPNP
        # )
        # result = list(pool.map(cv_pnp, zip(wld_chosed_all, obj_chosed_all)))
        # for retval, rvec, tvec, repro_error in result:
        #     for r_, t_, e_ in zip(rvec, tvec, repro_error):
        #         all_e.append(float(e_))
        #         all_q.append(r_)
        #         all_t.append(t_)

        for indice in itertools.combinations(range(len(obj_pts)), 4):
            indice = np.asarray(indice)
            obj_chosed, wld_chosed = obj_pts[indice], wld_pts[indice]

            retval, rvec, tvec, repro_error = cv2.solvePnPGeneric(
                wld_chosed, obj_chosed, Camera.K, Camera.dist, flags=cv2.SOLVEPNP_EPNP
            )
            for r_, t_, e_ in zip(rvec, tvec, repro_error):
                all_e.append(float(e_))
                all_q.append(r_)
                all_t.append(t_)

        index_top_K = np.argsort(all_e)[: self.topk]
        err_opt = 100000
        rvec_opt, tvec_opt = None, None

        repro_th = self.reprojectionError
        while rvec_opt is None:
            for index in index_top_K:
                r_, t_ = all_q[index], all_t[index]
                obj_rep, _ = cv2.projectPoints(wld_pts, r_, t_, Camera.K, Camera.dist)
                rep_err_before_lm = np.linalg.norm(obj_rep - obj_pts, axis=-1)
                rep_err_before_lm = rep_err_before_lm.squeeze()
                inlier_index = np.where(rep_err_before_lm < repro_th)[0]
                rep_err_before_lm = np.sum(rep_err_before_lm).item()
                if len(inlier_index) < 4:
                    repro_th += 5
                    # print(repro_th)
                    continue

                inlier_obj = obj_pts[inlier_index]
                inlier_wld = wld_pts[inlier_index]
                # rvec, tvec = cv2.solvePnPRefineLM(
                #     inlier_wld, inlier_obj, Camera.K, Camera.dist, r_, t_,
                # )
                rvec, tvec = ceres_pnp(inlier_wld, inlier_obj, r_, t_)

                obj_rep, _ = cv2.projectPoints(
                    wld_pts, rvec, tvec, Camera.K, Camera.dist
                )
                rep_err = np.linalg.norm(obj_rep - obj_pts, axis=-1).sum()
                if rep_err > rep_err_before_lm:
                    rvec, tvec = r_, t_
                    rep_err = rep_err_before_lm

                if err_opt > rep_err:
                    err_opt = rep_err
                    rvec_opt, tvec_opt = rvec, tvec

        Rmat = cv2.Rodrigues(rvec_opt)[0]
        quat = mathutils.Matrix(Rmat).to_quaternion()
        quat = np.asarray(quat)
        return quat.flatten(), tvec_opt.flatten()

    def __call__(self, points, logits, area):
        raise NotImplementedError


class Multi_Mean_PoseSolver(PoseSolver):
    def __init__(self, args):
        super(Multi_Mean_PoseSolver, self).__init__(args)

    def mean_and_filter(self, obj_pts):
        """每个控制点的所有预测像点求平均，并以sigma为准则剔除粗大误差"""
        result = {}
        for label, points in obj_pts.items():
            num_points = len(points)
            points = np.vstack(points)
            points_mean = np.mean(points, axis=0, keepdims=True)
            if num_points < 3:
                result[label] = points_mean.flatten()
            else:
                distances = cdist(points, points_mean).flatten()
                std_dist = np.sqrt(np.mean(distances**2))
                inlier_index = distances < std_dist * 3  # 3 sigma
                if not np.any(inlier_index):
                    print("{:d} points: all larger that 3 sigma".format(label))
                result[label] = np.mean(points[inlier_index], axis=0).flatten()
        return result

    def __call__(self, multi_points, multi_logits, multi_area):
        assert isinstance(multi_points, list)
        assert isinstance(multi_logits, list)
        assert len(multi_points) == len(multi_logits)
        area = multi_area[0]
        area_all = np.asarray(multi_area) - area
        assert np.all(area_all == 0), "area false"
        self.get_repro_th(area)

        obj_pts_original = defaultdict(list)
        for points, logits in zip(multi_points, multi_logits):
            labels, scores = self.find_index(logits)
            fg_indices = labels != logits.shape[1] - 1
            points, labels, scores = (
                points[fg_indices],
                labels[fg_indices],
                scores[fg_indices],
            )

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

        return self.solve_pnp(obj_pts, wld_pts)


class SimplePoseSolver(PoseSolver):
    """
    Solve pose according to the output from DETR
    """

    def __init__(self, args):
        super(SimplePoseSolver, self).__init__(args)

    def __call__(self, points, logits, area):
        """get quat and tra from the output of DETR
        Args:
            points: ndarray (num_queries x 2),
                    according to coordinates in original image
            logits: ndarray (num_queries x 12),
                    points classification results
            area: int, bounding box area
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

        assert points.shape[0] == logits.shape[0], "[Solver]: num_queries!"
        self.get_repro_th(area)

        labels, scores = self.find_index(logits)
        fg_indices = labels != logits.shape[1] - 1
        points, labels, scores = (
            points[fg_indices],
            labels[fg_indices],
            scores[fg_indices],
        )

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

        return self.solve_pnp(obj_pts, wld_pts)


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

    s_t = np.linalg.norm(t_pr - t_gt, ord=2) / np.linalg.norm(t_gt, ord=2)

    s_q = 2 * np.arccos(min(np.abs(np.dot(q_pr, q_gt)), 1))
    return s_t, s_q
