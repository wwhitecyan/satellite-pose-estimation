"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved https://github.com/facebookresearch/detr/blob/main/engine.py

by lyuwenyu
"""

import math
import os
import sys
import pathlib
from typing import Iterable

import torch
import torch.amp
import numpy as np
from src.data.speed import SpeedEval
from src.data import CocoEvaluator
from src.misc import MetricLogger, SmoothedValue, reduce_dict
import utils.misc as utils


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    **kwargs,
):
    writer = kwargs.get("tensorboard_writer", None)
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("class_error", SmoothedValue(fmt="{value:.3f}"))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = "Epoch: [{}]".format(epoch)
    print_freq = kwargs.get("print_freq", 10)

    ema = kwargs.get("ema", None)
    scaler = kwargs.get("scaler", None)
    i = 0

    dataset_len = len(data_loader.dataset)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        batch_size = len(samples)
        samples = samples.to(device)
        targets = [
            {k: v.to(device) if not isinstance(v, str) else v for k, v in t.items()}
            for t in targets
        ]
        # landmarkers = targets["landmarks"].to(device)
        i = i + 1

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

            class_error = loss_dict.pop("class_error")
            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets)

            loss_dict = criterion(outputs, targets)

            class_error = loss_dict.pop("class_error")
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        global_step = epoch * (dataset_len // batch_size) + i
        writer.add_scalar("train/class_error", class_error, global_step)
        writer.add_scalar("train/loss", loss.item(), global_step)
        writer.add_scalar("train/loss_ce", loss_dict["loss_ce"].item(), global_step)
        writer.add_scalar(
            "train/loss_point", loss_dict["loss_bbox"].item(), global_step
        )
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(class_error=class_error)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model,
    criterion,
    postprocessors,
    data_loader,
    gt_file,
    device,
    output_dir,
    index_file,
):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Test:"

    speed_evaluator = SpeedEval(gt_file, index_file)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        # samples shape (50,3,256,256)
        samples = samples.to(device)
        targets_fileanems = [item.pop("filename") for item in targets]
        targets_clip_bbox = [item.pop("clip_bbox") for item in targets]

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )

        metric_logger.update(class_error=loss_dict_reduced["class_error"])
        # aux_0 = np.asarray(outputs["aux_outputs"][0]["pred_pts"].cpu())
        # aux_1 = np.asarray(outputs["aux_outputs"][1]["pred_pts"].cpu())
        # aux_2 = np.asarray(outputs["aux_outputs"][2]["pred_pts"].cpu())
        aux_0 = np.asarray(outputs["aux_outputs"][0]["pred_logits"].cpu())
        aux_1 = np.asarray(outputs["aux_outputs"][1]["pred_logits"].cpu())
        aux_2 = np.asarray(outputs["aux_outputs"][2]["pred_logits"].cpu())
        # print("\n start aux_0:", aux_0, "\n \t aux_1:", aux_1, "\n \t aux_2:", aux_2)

        results = postprocessors(outputs, targets_clip_bbox)

        res = {filename: ret for filename, ret in zip(targets_fileanems, results)}
        res_aux0 = {filename: ret for filename, ret in zip(targets_fileanems, aux_0)}
        res_aux1 = {filename: ret for filename, ret in zip(targets_fileanems, aux_1)}
        res_aux2 = {filename: ret for filename, ret in zip(targets_fileanems, aux_2)}

        if speed_evaluator is not None:
            speed_evaluator.update(res, [res_aux0, res_aux1, res_aux2])
            # speed_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    speed_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if speed_evaluator is not None:
        stats["speed_eval_pose"] = speed_evaluator.stats
        print(speed_evaluator.stats)
    return stats, speed_evaluator
