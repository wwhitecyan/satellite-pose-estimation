#!/bin/sh
# File              : gen_single.sh
# Author            : WangZi <wangzitju@163.com>
# Date              : 03.12.2020
# Last Modified Date: 03.12.2020
# Last Modified By  : WangZi <wangzitju@163.com>
#python -m ipdb tools/train.py\
#    -c configs/rtdetr_speed/rtdetr_r18vd_6x_coco.yml\
#!/bin/bash

echo "训练kl 1to6"
#python -m ipdb tools/train.py -c configs/rtdetr_speed/rtdetr_mobilenetv3_6x_speed_kl_1.yml;
#python tools/train.py -c configs/rtdetr_speed/rtdetr_mobilenetv3_6x_speed_kl_2.yml;
python tools/train.py -c configs/rtdetr_speed/rtdetr_mobilenetv3_6x_speed_kl_1.yml;
