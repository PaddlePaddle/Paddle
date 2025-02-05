# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import random
import unittest

import numpy as np
from op_test import OpTest

import paddle
from paddle.base import core


def gen_input_help(input, rank_offset, max_rank, max_size):
    input_row, input_col = input.shape
    max_ins = np.max((max_size, input_row))
    input_help = np.zeros(max_ins * max_rank * input_col)
    ins_rank = np.zeros((max_ins, 1))
    ins_rank.fill(-1)

    output_col = max_rank * input_col
    output_row = input_row

    for idx in range(output_col * output_row):
        output_col_idx = idx % output_col
        output_row_idx = int(idx / output_col)
        k = int(output_col_idx / input_col)
        faster = rank_offset[output_row_idx, 2 * k + 1] - 1

        if output_col_idx == 0:
            ins_rank[output_row_idx] = rank_offset[output_row_idx, 0]

        if rank_offset[output_row_idx, 0] - 1 < 0 or faster < 0:
            continue

        rank_input_col_idx = output_col_idx % input_col
        index = rank_offset[output_row_idx, 2 * k + 2]
        input_help[idx] = input[index, rank_input_col_idx]
    input_help = input_help.reshape([max_ins, max_rank * input_col])

    return input_help, ins_rank


def gen_param_help(input, rank_offset, param, max_rank):
    input_row, input_col = input.shape
    rank_offset_row, rank_offset_col = rank_offset.shape
    param_row, param_col = param.shape

    block_matrix_row = input_col * max_rank

    output_param_row = block_matrix_row * input_row
    output_param_col = param_col

    output_param = np.zeros((output_param_row * output_param_col,))

    for idx in range(output_param_row * output_param_col):
        output_col_idx = idx % output_param_col
        output_row_idx = int(idx / output_param_col)
        ins_idx = int(output_row_idx / block_matrix_row)
        start_offset = output_row_idx % block_matrix_row
        k = int(start_offset / input_col)
        k_offset = start_offset % input_col

        lower = rank_offset[ins_idx, 0] - 1
        faster = rank_offset[ins_idx, 2 * k + 1] - 1
        if lower < 0 or faster < 0:
            continue
        start = lower * max_rank + faster
        ori_idx = (
            start * param_col * input_col
            + k_offset * param_col
            + output_col_idx
        )
        output_param[idx] = param[int(ori_idx / param_col), ori_idx % param_col]

    output_param = output_param.reshape([output_param_row, output_param_col])
    return output_param


def np_rank_attention(input, rank_offset, rank_para, max_rank, max_size):
    input_row, input_col = input.shape
    rank_offset_row, rank_offset_col = rank_offset.shape
    rank_para_row, rank_para_col = rank_para.shape

    assert input_row == rank_offset_row
    assert max_rank == ((rank_offset_col - 1) / 2)
    assert rank_para_row == max_rank * max_rank * input_col

    input_help, ins_rank = gen_input_help(
        input, rank_offset, max_rank, max_size
    )
    param_help = gen_param_help(input, rank_offset, rank_para, max_rank)
    block_matrix_row = input_col * max_rank

    res = np.zeros((input_row, rank_para_col))
    for ins in range(input_row):
        res[ins, :] = np.dot(
            input_help[ins, :],
            param_help[
                int(block_matrix_row * ins) : int(block_matrix_row * (ins + 1)),
                :,
            ],
        )
    return res, input_help, param_help, ins_rank


def gen_rank_offset(pv_nums, max_rank):
    all_ins_num = 0
    pv_rank_msg = []
    for _ in range(pv_nums):
        ins_pv = np.random.randint(1, max_rank + 2)  # 1~4
        rank_list = list(range(1, ins_pv + 1))
        random.shuffle(rank_list)
        all_ins_num = all_ins_num + ins_pv
        pv_rank_msg.append(rank_list)

    rank_offset = np.zeros((all_ins_num, max_rank * 2 + 1)).astype("int32")
    rank_offset.fill(-1)
    index = 0
    for pv_number in range(len(pv_rank_msg)):
        pv_ins = pv_rank_msg[pv_number]
        ad_num = len(pv_ins)
        index_start = index

        for j in range(ad_num):
            rank = -1
            if pv_ins[j] <= max_rank:
                rank = pv_ins[j]
            rank_offset[index, 0] = rank

            if rank > 0:
                for k in range(ad_num):
                    fast_rank = -1
                    if pv_ins[k] <= max_rank:
                        fast_rank = pv_ins[k]
                    if fast_rank > 0:
                        m = fast_rank - 1
                        rank_offset[index, 2 * m + 1] = pv_ins[k]
                        rank_offset[index, 2 * m + 2] = index_start + k
            index = index + 1
    return all_ins_num, rank_offset


def api_wrapper(x, rank_offset, rank_param, max_rank=3, max_size=0):
    ret = paddle._C_ops.rank_attention(
        x, rank_offset, rank_param, max_rank, max_size
    )
    return ret[1]


class TestRankAttentionOpComplex(OpTest):
    def config(self):
        self.pv_num = 100
        self.x_feat = 10
        self.y_feat = 15
        self.max_rank = 3
        self.dtype = "float64"

    def setUp(self):
        self.op_type = "rank_attention"
        self.python_api = api_wrapper
        self.python_out_sig = ['Out']
        self.config()
        ins_num, rank_offset = gen_rank_offset(self.pv_num, self.max_rank)
        input = np.random.random((ins_num, self.x_feat)).astype(self.dtype)
        rank_para_shape = [
            self.max_rank * self.max_rank * self.x_feat,
            self.y_feat,
        ]
        rank_para = np.random.random(rank_para_shape).astype(self.dtype)
        np_out, np_input_help, np_param_help, np_ins_rank = np_rank_attention(
            input,
            np.array(rank_offset),
            rank_para,
            self.max_rank,
            self.pv_num * 7,
        )
        self.inputs = {
            "X": input,
            "RankOffset": np.array(rank_offset).astype("int32"),
            "RankParam": rank_para,
        }
        self.attrs = {'MaxRank': self.max_rank, 'MaxSize': self.pv_num * 7}
        self.outputs = {
            "Out": np_out,
            "InputHelp": np_input_help,
            "InsRank": np_ins_rank,
        }

    def test_check_output_gpu(self):
        if core.is_compiled_with_cuda():
            self.check_output_with_place(core.CUDAPlace(0), check_pir=True)

    def test_check_grad_gpu(self):
        if core.is_compiled_with_cuda():
            self.check_grad_with_place(core.CUDAPlace(0), ["RankParam"], "Out")


class TestRankAttentionOpCpu(OpTest):
    def config(self):
        self.pv_num = 100
        self.x_feat = 10
        self.y_feat = 15
        self.max_rank = 3
        self.dtype = "float64"

    def setUp(self):
        self.op_type = "rank_attention"
        self.config()
        ins_num, rank_offset = gen_rank_offset(self.pv_num, self.max_rank)
        input = np.random.random((ins_num, self.x_feat)).astype(self.dtype)
        rank_para_shape = [
            self.max_rank * self.max_rank * self.x_feat,
            self.y_feat,
        ]
        rank_para = np.random.random(rank_para_shape).astype(self.dtype)
        np_out, np_input_help, np_param_help, np_ins_rank = np_rank_attention(
            input,
            np.array(rank_offset),
            rank_para,
            self.max_rank,
            self.pv_num * 7,
        )
        self.inputs = {
            "X": input,
            "RankOffset": np.array(rank_offset).astype("int32"),
            "RankParam": rank_para,
        }
        self.attrs = {'MaxRank': self.max_rank, 'MaxSize': self.pv_num * 7}
        self.outputs = {
            "Out": np_out,
            "InputHelp": np_input_help,
            "InsRank": np_ins_rank,
        }

    def test_check_output_cpu(self):
        try:
            self.check_output_with_place(place=core.CPUPlace())
        except:
            print("do not support cpu test, skip")

    def test_check_grad_cpu(self):
        try:
            self.check_grad_with_place(core.CPUPlace(), ["RankParam"], "Out")
        except:
            print("do not support cpu test, skip")


if __name__ == "__main__":
    unittest.main()
