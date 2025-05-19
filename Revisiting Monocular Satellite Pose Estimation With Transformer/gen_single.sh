#!/bin/sh
# File              : gen_single.sh
# Author            : WangZi <wangzitju@163.com>
# Date              : 03.12.2020
# Last Modified Date: 03.12.2020
# Last Modified By  : WangZi <wangzitju@163.com>
python gen_submission_single.py\
    --batch_size 1\
    --input_size 224 --batch_size 20\
    --backbone resnet50s8 --num_queries 30\
    --position_embedding sine\
    --enc_layers 4 --dec_layers 4\
    --resume ./work_dirs/train_ed4_resnet50s8/checkpoint.pth
