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

import test_op_transcriber

import paddle
from paddle.base.layer_helper import LayerHelper


class TestPullSparseV2OpTranscriber(test_op_transcriber.TestOpTranscriber):
    def append_op(self):
        self.op_type = "push_sparse_v2"
        ids = paddle.ones(shape=(100, 2, 3), dtype='float32')
        w = paddle.ones(shape=(100, 2, 3), dtype='float32')
        out = paddle.ones(shape=(100, 2, 3), dtype='float32')
        attrs = {
            'embeddingdim': 11,
            'tableid': 0,
            'accessorclass': "",
            'ctrlabelname': "",
            'paddingid': 0,
            'scalesparsegrad': True,
            'inputnames': [],
            'is_distributed': True,
        }
        helper = LayerHelper(self.op_type)
        helper.append_op(
            type=self.op_type,
            inputs={"Ids": ids, "W": w},
            outputs={"Out": out},
            attrs=attrs,
        )

    def test_translator(self):
        self.check()


if __name__ == "__main__":
    unittest.main()
