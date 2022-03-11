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
    complete_param = np.arange(64).reshape([8, 8])
    param_row = np.split(complete_param, 2, axis=0)
    param_col = np.split(complete_param, 2, axis=1)
    param_name = "param_0"
    complet_strategy = {
        param_name: {
            "process_shape": [2],
            "process_group": [0, 1],
            "dims_mapping": [-1, -1]
        }
    }
    row_strategy = {
        param_name: {
            "process_shape": [2],
            "process_group": [0, 1],
            "dims_mapping": [0, -1]
        }
    }
    col_strategy = {
        param_name: {
            "process_shape": [2],
            "process_group": [0, 1],
            "dims_mapping": [-1, 0]
        }
    }

    # test merge
    param_dict = {param_name: param_row}
    converter = Converter(param_dict, row_strategy, complet_strategy)
    convert_param_dict = converter.convert()
    assert np.equal(convert_param_dict[param_name], complete_param).all()

    # test slice
    param_dict = {param_name: [complete_param]}
    converter = Converter(param_dict, complet_strategy, col_strategy)
    convert_param_dict = converter.convert()
    assert np.equal(convert_param_dict[param_name], param_col[rank_id]).all()

    # test merge and slice
    param_dict = {param_name: param_col}
    converter = Converter(param_dict, col_strategy, row_strategy)
    convert_param_dict = converter.convert()
    assert np.equal(convert_param_dict[param_name], param_row[rank_id]).all()

    # test merge and slice with prefix match
    new_name = "param_1"
    row_strategy = {
        new_name: {
            "process_shape": [2],
            "process_group": [0, 1],
            "dims_mapping": [0, -1]
        }
    }
    converter = Converter(param_dict, col_strategy, row_strategy)
    convert_param_dict = converter.convert(strict=False)
    assert np.equal(convert_param_dict[new_name], param_row[rank_id]).all()


if __name__ == "__main__":
    test_convert()
