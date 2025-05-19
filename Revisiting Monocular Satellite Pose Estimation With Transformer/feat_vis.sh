#!/bin/bash
# File              : feat_vis.sh
# Author            : WangZi <wangzitju@163.com>
# Date              : 09.12.2020
# Last Modified Date: 09.12.2020
# Last Modified By  : WangZi <wangzitju@163.com>
python  visualize_features.py\
    --input_size 224\
    --backbone resnet50s8\
    --num_queries 30\
    --enc_layers 4 --dec_layers 4\
    --resume ./work_dirs/train_ed4_resnet50s8/checkpoint.pth\
    --img_path ./data/speed/images/real_test/img000111real.jpg\
    --bbox_path ./data/speed/annos/wz_real_test.json

