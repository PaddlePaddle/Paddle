# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import test_op_translator

import paddle
from paddle.base.framework import (
    convert_np_dtype_to_proto_type,
)
from paddle.base.layer_helper import LayerHelper

paddle.pir_utils._switch_to_old_ir_()


class TestDistributedPushSparseOpTranslator(
    test_op_translator.TestOpTranslator
):
    def append_op(self):
        self.op_type = "distributed_push_sparse"
        ids = paddle.ones(shape=(1, 1), dtype='float32')
        shows = paddle.ones(shape=(1, 1), dtype='float32')
        clicks = paddle.ones(shape=(1, 1), dtype='float32')
        output = paddle.ones(shape=(1, 1), dtype='float32')
        attrs = {
            'table_id': 0,
            'size': 8,
            'is_distributed': False,
            'push_sparse_version': 'push_sparse',
            'padding_idx': -1,
            'dtype': convert_np_dtype_to_proto_type(np.float32),
            'is_test': False,
            'use_cvm_op': False,
        }
        helper = LayerHelper(self.op_type)
        helper.append_op(
            type=self.op_type,
            inputs={"Ids": [ids], "Shows": shows, "Clicks": clicks},
            outputs={"Outputs": [output]},
            attrs=attrs,
        )

    def test_translator(self):
        self.check()


if __name__ == "__main__":
    unittest.main()
