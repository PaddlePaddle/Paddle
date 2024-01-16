# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import test_op_translator

import paddle
from paddle.base.layer_helper import LayerHelper


class TestDistributedPushSparseOpTranslator(
    test_op_translator.TestOpTranslator
):
    def append_op(self):
        self.op_type = "distributed_push_sparse"
        ids = paddle.ones(shape=(100, 2, 3), dtype='float32')
        shows = paddle.ones(shape=(100, 2, 3), dtype='float32')
        clicks = paddle.one(shape=(100, 2, 3), dtype='float32')
        out = paddle.ones(shape=(100, 2, 3), dtype='float32')
        attrs = {
            'table_id': 0,
            'size': 0,
            'is_distributed': False,
            'push_sparse_version': "push_sparse",
            'padding_idx': -1,
            'dtype': "float32",
            'is_test': False,
            'use_cvm_op': False,
        }
        helper = LayerHelper(self.op_type)
        helper.append_op(
            type=self.op_type,
            inputs={"Ids": ids, "Shows": shows, "Clicks": clicks},
            outputs={"Outputs": out},
            attrs=attrs,
        )

    def test_translator(self):
        self.check()


if __name__ == "__main__":
    unittest.main()
