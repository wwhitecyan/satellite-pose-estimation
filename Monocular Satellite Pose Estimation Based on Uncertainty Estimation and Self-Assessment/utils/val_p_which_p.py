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
from tqdm import tqdm
import json
import random
import numpy as np
from utils import Camera
from collections import defaultdict
import cv2
import mathutils
from scipy.spatial.distance import cdist
import PyCeres


def build_solver():
    return SimplePoseSolver()


def build_sigma_solver(num_points_PNP):
    return SimplePoseSolverSigma(num_points_PNP)


world_pt_path = "/media/willer/ST1/datasets/speed/annos/all_result.json"
# world_pt_path = "/media/willer/ST1/datasets/speed/speed/annos/all_result_del_0_1.json"


def read_GT(file_path):
    # 读取真值的json文件
    with open(file_path, "r") as f:
        bbox_ann = json.load(f)
    landmarks = {item["filename"]: item["landmarks"] for item in bbox_ann}
    # {'img000001.jpg': [[x1, y1, x2, y2], [[x1, y1], [x2, y2], ... ]];  }
    bbox = {item["filename"]: item["bbox_xxyy"] for item in bbox_ann}

    return landmarks, bbox


def read_predict(file_path):
    # 读取预测结果的json文件
    with open(file_path, "r") as f:
        predict_ann = json.load(f)
    result = predict_ann.items()
    # print(dir(predict_ann.items()))

    predict_ann = {
        key: [
            value["points"],
            value["logits"],
            value["sigma"],
            value["score"],
            value["quat_gt"],
            value["tvec_gt"],
        ]
        for key, value in predict_ann.items()
    }

    return predict_ann


class PoseSolver(object):
    """
    Solve pose according to the output from DETR
    """

    def __init__(self):
        with open(world_pt_path, "r") as f:
            world_pt = json.load(f)
        W_Pt = []
        for item in world_pt:
            W_Pt.append(item["pt"])
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
                std_dist = np.std(distances)
                inlier_index = distances < std_dist * 3  # 3 sigma
                if not np.all(inlier_index):
                    print("{:d} points: all larger that 3 sigma".format(label))
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
            # quat = quaternions.quaternion_from_matrix(np.linalg.inv(Rmat))
            quat = np.asarray(quat)
        except cv2.error as e:
            raise e

        if np.any(np.isnan(quat)):
            print(rvec, quat)

        return quat.flatten(), tvec.flatten()


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
    return logits.argmax(1), logits.max(1)


class SimplePoseSolver(PoseSolver):
    """
    Solve pose according to the output from DETR
    """

    def __init__(self):
        super(SimplePoseSolver, self).__init__()
        self.reprojectionError = 25

    def __call__(self, points, logits):
        """get quat and tra from the output of DETR
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

        assert points.shape[0] == logits.shape[0], "[Solver]: num_queries!"

        # labels.shape 30 scores.shape 30
        labels, scores = find_index(logits)
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
            # TODO choose points with entropy
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
            # wld_pts.shape 11,1,3 obj_pts.shape 11,1,2
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
            # quat = quaternions.quaternion_from_matrix(np.linalg.inv(Rmat))
            quat = np.asarray(quat)
        except cv2.error as e:
            raise e

        if np.any(np.isnan(quat)):
            print(rvec, quat)

        return quat.flatten(), tvec.flatten()


def ceres_pnp(wld_pts, obj_pts, sigma, rvec, tvec):
    """
    obj_pts: Nx1x2
    wld_pts: Nx1x3
    """
    obj_pts = cv2.undistortPoints(obj_pts, Camera.K, Camera.dist)
    obj_pts, wld_pts, sigma = (
        np.squeeze(obj_pts),
        np.squeeze(wld_pts),
        np.squeeze(sigma),
    )
    num_points = len(obj_pts)
    camera = np.concatenate((rvec.flatten(), tvec.flatten()))

    # print(sigma)
    # sigma[sigma < 1] = 1
    sigma = np.sqrt(sigma)
    sigma_w1 = 1 / (sigma + 1e-6)
    sigma_sum = np.sum(sigma_w1, axis=0)
    sigma_w2 = sigma_w1 / sigma_sum
    # sigma_w3 = sigma_w1.copy()
    # sigma_w3[:, 0] = sigma_w3[:, 0] / sigma_sum[0]
    # sigma_w3[:, 1] = sigma_w3[:, 1] / sigma_sum[1]

    problem = PyCeres.Problem()
    for idx in range(num_points):
        loss = PyCeres.HuberLoss(0.005)

        cost_function = PyCeres.CreatePnPCostFunction(
            obj_pts[idx, 0],
            sigma_w2[idx, 0],
            obj_pts[idx, 1],
            sigma_w2[idx, 1],
            wld_pts[idx, 0],
            wld_pts[idx, 1],
            wld_pts[idx, 2],
            1.0,
            0.0,
        )
        problem.AddResidualBlock(cost_function, loss, camera)

    options = PyCeres.SolverOptions()
    options.linear_solver_type = PyCeres.LinearSolverType.DENSE_QR
    options.max_num_iterations = 20
    # options.minimizer_progress_to_stdout = True

    summary = PyCeres.Summary()
    PyCeres.Solve(options, problem, summary)
    # print(summary.FullReport())
    # print(camera)
    return np.asarray(camera[:3]), np.asarray(camera[3:])


class SimplePoseSolverSigma(PoseSolver):
    """
    Solve pose according to the output from DETR
    """

    def __init__(self, num_points_PNP):
        super(SimplePoseSolverSigma, self).__init__()
        # self.reprojectionError = args.repro
        self.reprojectionError = 25
        self.num_points_PNP = num_points_PNP

    def __call__(self, points, logits, sigmas):
        """get quat and tra from the output of DETR
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

        assert points.shape[0] == logits.shape[0], "[Solver]: num_queries!"

        labels, scores = find_index(logits)
        fg_indices = labels != logits.shape[1] - 1
        points, labels, scores, sigmas = (
            points[fg_indices],
            labels[fg_indices],
            scores[fg_indices],
            sigmas[fg_indices],
        )

        P = defaultdict(list)
        for pt, l_, s_, sig in zip(points, labels, scores, sigmas):
            l_ = int(l_)
            P[int(l_)].append([pt[0], pt[1], s_, sig[0], sig[1]])

        obj_pts, wld_pts, sig_pts = [], [], []
        for l_, pt_score in P.items():
            pt_score = np.asarray(pt_score)
            max_idx = pt_score[:, 2].argmax()
            pts = pt_score[max_idx]
            wld_pts.append(self.W_Pt[l_])
            obj_pts.append(pts[:2])
            sig_pts.append(pts[3:5])

        try:
            wld_pts = np.asarray(wld_pts)[:, np.newaxis, :].astype(np.float32)
            obj_pts = np.asarray(obj_pts)[:, np.newaxis, :].astype(np.float32)
            sig_pts = np.asarray(sig_pts)[:, np.newaxis, :].astype(np.float32)
        except IndexError as e:
            raise e

        try:
            test_P_num_p = True
            if test_P_num_p:
                if len(wld_pts) == 11:
                    random_index = random.sample(range(11), self.num_points_PNP)
                    wld_pts = wld_pts[random_index]
                    obj_pts = obj_pts[random_index]
                    sig_pts = sig_pts[random_index]
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                wld_pts,
                obj_pts,
                Camera.K,
                Camera.dist,
                useExtrinsicGuess=False,
                flags=cv2.SOLVEPNP_EPNP,
                reprojectionError=self.reprojectionError,
            )

            if np.any(np.isnan(rvec)):
                print(rvec)

            if inliers is not None:
                rvec, tvec = ceres_pnp(
                    wld_pts[inliers.flatten()],
                    obj_pts[inliers.flatten()],
                    sig_pts[inliers.flatten()],
                    rvec,
                    tvec,
                )

            Rmat = cv2.Rodrigues(rvec)[0]
            quat = mathutils.Matrix(Rmat).to_quaternion()
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

    s_t = np.linalg.norm(t_pr - t_gt, ord=2) / np.linalg.norm(t_gt, ord=2)

    s_q = 2 * np.arccos(min(np.abs(np.dot(q_pr, q_gt)), 1))
    return s_t, s_q


if __name__ == "__main__":
    # read Predict and GT
    GT, bbox = read_GT("../plot/input_json/wz_train.json")
    predict = read_predict(
        "../output/rtdetr_mobilenetv3_6x_speed_6_kl_Large/eval_0191_log.json"
    )
    for ii in range(4, 12):
        s_all_list = []
        solver = build_sigma_solver(ii)
        for filename, predict_item in tqdm(predict.items()):
            # 遍历预测结果
            # landmarks = GT[filename]
            bounding_box = bbox[filename]
            pr_pts, pr_logits, pr_sigma, pr_score, quat_gt, tvec_gt = predict_item
            q_pr, t_pr = solver(pr_pts, pr_logits, np.asarray(pr_sigma))
            s_t, s_q = speed_score(q_pr, t_pr, quat_gt, tvec_gt)
            s_all_list.append(s_q + s_t)
        print(ii, np.mean(np.asarray(s_all_list)), "\n")
