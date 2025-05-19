"""
by lyuwenyu
"""
import time
import json
import datetime

import torch

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .speed_engine import train_one_epoch, evaluate
from utils.speed_eval import build_solver, SimplePoseSolver

# from tensorboard.summary import Writer
from tensorboardX import SummaryWriter

writer = SummaryWriter()

import re

import torch
import torch.nn as nn
import numpy as np


def count_conv_flops(module, input, output):
    # 获取卷积层的参数
    in_channels = module.in_channels
    out_channels = module.out_channels
    kernel_size = module.kernel_size
    output_size = output.size()

    # 计算 FLOPs
    flops = (
        in_channels
        * out_channels
        * kernel_size[0]
        * kernel_size[1]
        * output_size[2]
        * output_size[3]
    )
    return flops


def count_linear_flops(module, input, output):
    # 获取全连接层的参数
    in_features = module.in_features
    out_features = module.out_features

    # 计算 FLOPs
    flops = in_features * out_features
    return flops


def count_flops(model, input_size,device):
    # 注册钩子函数
    flops = 0

    def conv_hook(module, input, output):
        nonlocal flops
        flops += count_conv_flops(module, input, output)

    def linear_hook(module, input, output):
        nonlocal flops
        flops += count_linear_flops(module, input, output)

    hooks = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(conv_hook))
        elif isinstance(layer, nn.Linear):
            hooks.append(layer.register_forward_hook(linear_hook))

    # 前向传播以触发钩子
    input_tensor = torch.randn(*input_size).to(device)
    model(input_tensor)

    # 移除钩子
    for hook in hooks:
        hook.remove()

    # 返回 GFLOPs
    return flops / 1e9


class SpeedSolver(BaseSolver):
    def fit(
        self,
    ):
        print("Start training")
        self.train()

        args = self.cfg

        n_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print("number of params:", n_parameters)

        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
        best_stat = {
            "epoch": -1,
        }
        tf_writer = None

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                args.clip_max_norm,
                print_freq=args.log_step,
                ema=self.ema,
                scaler=self.scaler,
                tensorboard_writer=writer,
            )

            self.lr_scheduler.step()

            if self.output_dir:
                checkpoint_paths = [self.output_dir / "checkpoint.pth"]
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_paths.append(
                        self.output_dir / f"checkpoint{epoch:04}.pth"
                    )
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            # module = self.ema.module if self.ema else self.model
            # test_stats, coco_evaluator = evaluate(
            #     module,
            #     self.criterion,
            #     self.postprocessor,
            #     self.val_dataloader,
            #     self.cfg.yaml_cfg["val_dataloader"]["dataset"]["ann_file"],
            #     self.device,
            #     build_solver(),
            #     self.output_dir,
            # )
            if epoch > -1:
                self.eval()

                # base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)

                module = self.ema.module if self.ema else self.model
                test_stats, coco_evaluator = evaluate(
                    module,
                    self.criterion,
                    self.postprocessor,
                    self.val_dataloader,
                    self.cfg.yaml_cfg["val_dataloader"]["dataset"]["ann_file"],
                    self.device,
                    self.output_dir,
                    self.cfg.yaml_cfg["val_dataloader"]["dataset"]["index_file"],
                )
                writer.add_scalar("eval/class_error", test_stats["class_error"], epoch)
                writer.add_scalar("eval/loss", test_stats["loss"], epoch)
                writer.add_scalar("eval/loss_ce", test_stats["loss_ce"], epoch)
                writer.add_scalar("eval/loss_point", test_stats["loss_bbox"], epoch)
                match_t = re.search(
                    r"tvec score: ([\d.]+)", test_stats["speed_eval_pose"]
                )
                if match_t:
                    tvec_score = float(match_t.group(1))
                    writer.add_scalar("eval/tvec_score", tvec_score, epoch)

                match_q = re.search(
                    r"quat score: ([\d.]+)", test_stats["speed_eval_pose"]
                )
                if match_q:
                    quat_score = float(match_q.group(1))
                    writer.add_scalar("eval/quat_score", quat_score, epoch)
                match_final_score = re.search(
                    r"final score: ([\d.]+)", test_stats["speed_eval_pose"]
                )
                if match_final_score:
                    final_score = float(match_final_score.group(1))
                    writer.add_scalar("eval/final_score", final_score, epoch)

                # # TODO
                # for k in test_stats.keys():
                #     if k in best_stat:
                #         best_stat["epoch"] = (
                #             epoch if test_stats[k][0] > best_stat[k] else best_stat["epoch"]
                #         )
                #         best_stat[k] = max(best_stat[k], test_stats[k][0])
                #     else:
                #         best_stat["epoch"] = epoch
                #         best_stat[k] = test_stats[k][0]
                # print("best_stat: ", best_stat)

                log_stats = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    **{f"test_{k}": v for k, v in test_stats.items()},
                    "epoch": epoch,
                    "n_parameters": n_parameters,
                }

                if self.output_dir and dist.is_main_process():
                    with (self.output_dir / "log.txt").open("a") as f:
                        f.write(json.dumps(log_stats) + "\n")

                    with open(f"{self.output_dir}/eval_{epoch:04d}_log.json", "w") as f:
                        json.dump(coco_evaluator.log, f)
                #     # for evaluation logs
            #     if coco_evaluator is not None:
            #         (self.output_dir / "eval").mkdir(exist_ok=True)
            #         if "bbox" in coco_evaluator.coco_eval:
            #             filenames = ["latest.pth"]
            #             if epoch % 50 == 0:
            #                 filenames.append(f"{epoch:03}.pth")
            #             for name in filenames:
            #                 torch.save(
            #                     coco_evaluator.coco_eval["bbox"].eval,
            #                     self.output_dir / "eval" / name,
            #                 )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))
        tensorboard_path = [self.output_dir / "tensorboard.json"]
        writer.export_scalars_to_json(tensorboard_path[0])
        writer.close()

    def val(
        self,
    ):
        self.eval()

        # base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.cfg.yaml_cfg["val_dataloader"]["dataset"]["ann_file"],
            self.device,
            self.output_dir,
            self.cfg.yaml_cfg["val_dataloader"]["dataset"]["index_file"],
        )

        # 定义输入大小（假设输入为 256x256 的 RGB 图像）
        input_size = (2, 3, 256, 256)

        # 计算 GFLOPs
        gflops = count_flops(module, input_size,self.device)
        print(f"Model GFLOPs: {gflops:.2f}")

        if self.output_dir:
            with open(f"{self.output_dir}/eval_log.json", "w") as f:
                json.dump(coco_evaluator.log, f)
            # dist.save_on_master(
            # coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
            #     self.output_dir / "eval.pth",
            # )

        return
