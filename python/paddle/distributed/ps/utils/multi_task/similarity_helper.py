# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multi-Task utils."""

import numpy as np
import os
from math import sqrt
import warnings

__all__ = []


class TaskSimilarityAnalyse:
    """
    multi-task's similarity analyse tool.
    """

    def __init__(self, data_path=None):
        assert data_path != None, "data_path is missed!"
        self.data_path = data_path
        self.task_name2id = {}
        self.data_read()

    #element multiple and sum
    def multipl(self, a, b):
        sumofab = 0.0
        for i in range(len(a)):
            temp = a[i] * b[i]
            sumofab += temp
        return sumofab

    #calculate relation of 2 in pearson
    def relation_cal_2(self, x, y):
        n = len(x)
        #sum
        sum1 = sum(x)
        sum2 = sum(y)
        #element multipl and sum
        sumofxy = self.multipl(x, y)
        #square sum
        sumofx2 = sum([pow(i, 2) for i in x])
        sumofy2 = sum([pow(j, 2) for j in y])
        num = sumofxy - (float(sum1) * float(sum2) / n)
        #pearson
        den = sqrt(
            (sumofx2 - float(sum1**2) / n) * (sumofy2 - float(sum2**2) / n))
        return num / den

    #in case that data's length is different
    def window_smooth(sefl, x, y):
        x_len = len(x)
        y_len = len(y)
        if x_len == y_len:
            return self.relation_cal_2(x, y)
        max_ = x
        min_ = y
        if y_len > x_len:
            max_ = y
            min_ = x
        max_score = 0
        min_len = len(min_)
        max_len = len(max_)
        for i in range(max_len - min_len):
            tmp_slice = slice(i, i + min_len)
            tmp_score = relation_cal_2(max_[tmp_slice], min_)
            max_score = max(max_score, tmp_score)
        return max_score

    #calculate tasks relation in pearson
    def get_similarity_score(self, data):
        file_list = data
        file_num = len(file_list)
        result_sum = 0
        pair_sum = 0
        for i in range(file_num):
            for j in range(i + 1, file_num):
                result_sum += self.window_smooth(file_list[i], file_list[j])
                pair_sum += 1

        return float(result_sum / pair_sum)

    #read the label data
    def data_read(self):
        file_path = self.data_path
        dirs = os.listdir(file_path)
        file_list = []
        step = 0
        for file in dirs:
            f_l = []
            dir = os.path.join(file_path, file)
            with open(dir, 'r') as f:
                for line in f:
                    label = line.strip()
                    f_l.append(float(label))
            f.close()
            file_list.append(f_l)
            self.task_name2id[step] = str(file)
            step += 1
        self.file_list = file_list

    #get topk similarity tasks
    def get_topk(self, k):
        all_comb = []
        task_num = len(self.file_list)

        #get all the combination of k tasks
        def comb(sofar, rest, n):
            if n == 0:
                all_comb.append(sofar)
            else:
                for i in range(len(rest)):
                    comb(sofar + rest[i], rest[i + 1:], n - 1)

        task_num_l = [str(i) for i in range(task_num)]
        comb("", "".join(task_num_l), k)

        for i in range(len(all_comb)):
            all_comb[i] = list(all_comb[i])

        high_score_comb_no = 0
        high_score = 0

        for i in range(len(all_comb)):
            task_comb = []
            for index in all_comb[i]:
                task_comb.append(self.file_list[int(index)])
            result = self.get_similarity_score(task_comb)
            if result > high_score:
                high_score = result
                high_score_comb_no = i

        task_name_res = []

        for i in all_comb[high_score_comb_no]:
            task_name_res.append(self.task_name2id[int(i)])

        return task_name_res
