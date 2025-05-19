#!/bin/bash
# File              : get_backbone_time.sh
# Author            : WangZi <wangzitju@163.com>
# Date              : 13.12.2020
# Last Modified Date: 13.12.2020
# Last Modified By  : WangZi <wangzitju@163.com>
echo -------resnet50s16--------
python get_backbone_time.py\
    --input_size 448\
    --backbone resnet50\
    --test_num 200
echo -------resnet50s8--------
python get_backbone_time.py\
    --input_size 224\
    --backbone resnet50s8
    --test_num 200
