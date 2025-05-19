#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : label_colors.py
# Author            : WangZi <wangzitju@163.com>
# Date              : 09.12.2020
# Last Modified Date: 09.12.2020
# Last Modified By  : WangZi <wangzitju@163.com>
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from utils.utils import COLORS

points = (
    [1, 4], [2, 4], [3, 4],
    [1, 3], [2, 3], [3, 3],
    [1, 2], [2, 2], [3, 2],
    [1, 1], [2, 1],
)

for idx, pt in enumerate(points):
    colors = [item / 255 for item in COLORS[idx]]
    plt.plot(pt[0], pt[1], '.',  markersize=42, color=colors)
    plt.legend('{:02d}'.format(idx))

plt.axis('off')
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# for idx, pt in enumerate(points):
#     colors = [item / 255 for item in COLORS[idx]]
#     circle = mpatches.Circle(xy=pt, radius=0.1, facecolor=colors)
#     ax.add_patch(circle)

# plt.show()

# # create some data
# data = np.random.randint(0, 8, (5, 5))
# # get the unique values from data
# # i.e. a sorted list of all values in data
# values = np.unique(data.ravel())

# plt.figure(figsize=(8, 4))
# im = plt.imshow(data, interpolation='none')

# # get the colors of the values, according to the
# # colormap used by imshow
# colors = [im.cmap(im.norm(value)) for value in values]
# # create a patch (proxy artist) for every color
# patches = [mpatches.Patch(color=colors[i], label="Level {l}".format(
#     l=values[i])) for i in range(len(values))]
# # put those patched as legend-handles into the legend
# plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# plt.grid(True)
# plt.show()
