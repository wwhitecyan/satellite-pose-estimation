#!/bin/sh
# File              : gen_single.sh
# Author            : WangZi <wangzitju@163.com>
# Date              : 03.12.2020
# Last Modified Date: 03.12.2020
# Last Modified By  : WangZi <wangzitju@163.com>

echo "参数一为配置文件路径，参数二为checkpoint路径"
#python -m ipdb tools/train.py -c configs/rtdetr_speed/rtdetr_r18vd_6x_speed_kl_1.yml -r output/rtdetr_r18vd_6x_speed_1_kl/checkpoint0095.pth --test-only
python tools/train.py -c configs/rtdetr_speed/rtdetr_mobilenetv3_6x_speed_kl_1.yml -r output/rtdetr_mobilenetv3_6x_speed_1_kl_Large/checkpoint0191.pth --test-only
#python -m ipdb tools/train.py -c configs/rtdetr_speed/rtdetr_mobilenetv3_6x_speed_1.yml -r output/rtdetr_mobilenetv3_6x_speed_1_Large/checkpoint0191.pth --test-only
