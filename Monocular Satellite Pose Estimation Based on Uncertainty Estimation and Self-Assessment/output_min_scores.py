import numpy as np
import json
import os


def calculate_mean_std(data):
    mean = np.mean(data)
    std = np.std(data)
    print("mean:", mean)
    print("std:", std)


def print_output_min_scores(dir):
    scores_all = []
    t_scores_all = []
    q_scores_all = []

    # 遍历文件夹下的所有文件
    for filename in sorted(os.listdir(dir)):
        if filename.endswith(".json") and filename.startswith("eval_00"):
            # print(filename)
            file_path = os.path.join(dir, filename)
            scores_img = []
            t_scores_img = []
            q_scores_img = []
            # 读取 JSON 文件
            with open(file_path, "r") as file:
                data = json.load(file)
            for keys, value in data.items():
                scores_img.append(value["score"])
                t_scores_img.append(value["score_tvec"])
                q_scores_img.append(value["score_quat"])
            if len(scores_img) != 0:
                scores_all.append(sum(scores_img) / len(scores_img))
                t_scores_all.append(sum(t_scores_img) / len(t_scores_img))
                q_scores_all.append(sum(q_scores_img) / len(q_scores_img))

    # 输出包含 'score' 值的列表
    print("\n dir:", dir)
    print("\t final_score: {:.5f}".format(min(scores_all)))
    min_index = scores_all.index(min(scores_all))
    print("\t epoch: {:d}".format(min_index))
    print("\t t_score: {:.5f}".format(t_scores_all[min_index]))
    print("\t q_score: {:.5f}".format(q_scores_all[min_index]))


dir = "./output/"
for filename in sorted(os.listdir(dir)):
    subdir = os.path.join(dir, filename)
    print_output_min_scores(subdir)
