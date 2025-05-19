#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : get_runtime.py
# Author            : WangZi <wangzitju@163.com>
# Date              : 13.12.2020
# Last Modified Date: 13.12.2020
# Last Modified By  : WangZi <wangzitju@163.com>
"""
测试网络运行的时间
python gen_inference_time.py
"""
import numpy as np
import re


# 这里是从gen_submission_single.py输出的运行结果
result = """# noqa: E501
model_time: 0.101423  solver_time: 0.007778  time: 0.1937  data: 0.0781  max mem: 1002
model_time: 0.102917  solver_time: 0.011022  time: 0.1620  data: 0.0470  max mem: 1002
model_time: 0.101218  solver_time: 0.007785  time: 0.1614  data: 0.0469  max mem: 1002
model_time: 0.102279  solver_time: 0.007628  time: 0.1606  data: 0.0463  max mem: 1002
model_time: 0.102974  solver_time: 0.007766  time: 0.1586  data: 0.0441  max mem: 1002
model_time: 0.101548  solver_time: 0.007825  time: 0.1606  data: 0.0460  max mem: 1002
model_time: 0.102556  solver_time: 0.012218  time: 0.1595  data: 0.0451  max mem: 1002
model_time: 0.101677  solver_time: 0.008025  time: 0.1589  data: 0.0447  max mem: 1002
model_time: 0.101336  solver_time: 0.007844  time: 0.1584  data: 0.0448  max mem: 1002
model_time: 0.101448  solver_time: 0.008836  time: 0.1555  data: 0.0417  max mem: 1002
model_time: 0.102686  solver_time: 0.010328  time: 0.1556  data: 0.0412  max mem: 1002
model_time: 0.101651  solver_time: 0.007744  time: 0.1605  data: 0.0462  max mem: 1002
model_time: 0.103374  solver_time: 0.009768  time: 0.1626  data: 0.0490  max mem: 1002
model_time: 0.103537  solver_time: 0.009430  time: 0.1590  data: 0.0448  max mem: 1002
"""


content = []
for item in result.split('\n'):
    item = item.strip()
    if len(item) > 0:
        if item[0] != '#':
            content.append(item)

model_time, solver_time, all_time, data_time, mem = [[]]*5
all_times = []
for item in content:
    number_all = re.findall(r"(?<![a-zA-Z:])[-+]?\d*\.?\d+", item)
    all_times.append(number_all)
all_times = np.asarray(all_times).astype(np.float)

# 去掉第一行，dataloader时间不准
time_mean = np.mean(all_times[1:], 0)
model_mean, solver_mean, all_mean, data_mean, mem_mean = time_mean.tolist()
print(f'model: {model_mean:.6f}, solver: {solver_mean:.6f}, '
      f'data: {data_mean:.6f}, all: {all_mean:.6f}, mem: {mem_mean:.1f}')
