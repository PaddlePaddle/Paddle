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

import unittest
import numpy as np

import paddle
from paddle.distributed.auto_parallel.converter import Converter


def test_convert():
    rank_id = paddle.distributed.get_rank()
    complete_tensor = np.arange(64).reshape([8, 8])
    tensor_row = np.split(complete_tensor, 2, axis=0)
    tensor_col = np.split(complete_tensor, 2, axis=1)
    tensor_name = "tensor_0"
    complet_strategy = {
        tensor_name: {
            "process_shape": [2],
            "process_group": [0, 1],
            "dims_mapping": [-1, -1]
        }
    }
    row_strategy = {
        tensor_name: {
            "process_shape": [2],
            "process_group": [0, 1],
            "dims_mapping": [0, -1]
        }
    }
    col_strategy = {
        tensor_name: {
            "process_shape": [2],
            "process_group": [0, 1],
            "dims_mapping": [-1, 0]
        }
    }

    # test merge
    tensor_dict = {tensor_name: tensor_row}
    converter = Converter(tensor_dict, row_strategy, complet_strategy)
    convert_tensor_dict = converter.convert()
    assert np.equal(convert_tensor_dict[tensor_name], complete_tensor).all()

    # test slice
    tensor_dict = {tensor_name: [complete_tensor]}
    converter = Converter(tensor_dict, complet_strategy, col_strategy)
    convert_tensor_dict = converter.convert()
    assert np.equal(convert_tensor_dict[tensor_name], tensor_col[rank_id]).all()

    # test merge and slice
    tensor_dict = {tensor_name: tensor_col}
    converter = Converter(tensor_dict, col_strategy, row_strategy)
    convert_tensor_dict = converter.convert()
    assert np.equal(convert_tensor_dict[tensor_name], tensor_row[rank_id]).all()

    # test merge and slice with prefix match
    new_name = "tensor_1"
    row_strategy = {
        new_name: {
            "process_shape": [2],
            "process_group": [0, 1],
            "dims_mapping": [0, -1]
        }
    }
    converter = Converter(tensor_dict, col_strategy, row_strategy)
    convert_tensor_dict = converter.convert(strict=False)
    assert np.equal(convert_tensor_dict[new_name], tensor_row[rank_id]).all()

    # test sliced_shape is 1
    complete_tensor = np.arange(4).reshape([2, 2])
    tensor_row = np.split(complete_tensor, 2, axis=0)
    complet_strategy = {
        "tensor_2": {
            "process_shape": [2],
            "process_group": [0, 1],
            "dims_mapping": [-1, -1]
        }
    }
    row_strategy = {
        "tensor_2": {
            "process_shape": [2],
            "process_group": [0, 1],
            "dims_mapping": [0, -1]
        }
    }
    tensor_dict = {"tensor_2": [complete_tensor]}
    converter = Converter(tensor_dict, complet_strategy, row_strategy)
    convert_tensor_dict = converter.convert()
    assert np.equal(convert_tensor_dict["tensor_2"], tensor_row[rank_id]).all()


if __name__ == "__main__":
    test_convert()
