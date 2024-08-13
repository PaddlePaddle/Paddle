# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import json
import os
import tempfile
import unittest

import numpy as np
from op_test import OpTest

import paddle


def transpose_layout(x, src_layout, dst_layout):
    return x.transpose([0, 2, 3, 1])


class TestTransferLayoutFP16Op(OpTest):
    def setUp(self):
        paddle.enable_static()
        self.op_type = 'transfer_layout'
        self.dtype = np.float16
        x = np.random.random(size=[2, 5, 10, 10])
        self.inputs = {'X': x.astype(self.dtype)}
        self.outputs = {'Out': x.transpose([0, 2, 3, 1])}
        self.attrs = {'src_layout': 0, 'dst_layout': 1}
        self.python_api = transpose_layout

    def test_check_output(self):
        self.check_output()


class LayoutAutoTune(unittest.TestCase):
    def test_config(self):
        paddle.base.core.enable_layout_autotune()
        if self.use_autotune():
            self.assertEqual(paddle.base.core.use_layout_autotune(), True)
            paddle.base.core.disable_layout_autotune()
        self.assertEqual(paddle.base.core.use_layout_autotune(), False)
        self.use_autotune()

    def setUp(self):
        paddle.disable_static()
        self.use_autotune()

    def use_autotune(self):
        if paddle.is_compiled_with_cuda():
            paddle.incubate.autotune.set_config(
                config={"layout": {"enable": True}}
            )
            return paddle.base.core.use_layout_autotune()
        else:
            config = {"layout": {"enable": False}}
            tfile = tempfile.NamedTemporaryFile(mode="w+", delete=False)
            json.dump(config, tfile)
            tfile.close()
            paddle.incubate.autotune.set_config(tfile.name)
            os.remove(tfile.name)
            return paddle.base.core.use_layout_autotune()

    def test_flatten_op_transposer(self):
        conv = paddle.nn.Conv2D(3, 8, (3, 3))
        flatten = paddle.nn.Flatten(start_axis=1, stop_axis=2)
        data = paddle.rand([1, 3, 16, 14])
        with paddle.amp.auto_cast(level="O2"):
            conv_out = conv(data)
            # conv_out.shape = [1, 14, 12, 8] with NHWC
            # layout tuner will transpose conv_out to
            # [1, 8, 14, 12] with NCHW before the following flatten op
            # because it flatten the C and H dimensions.
            out = flatten(conv_out)

        self.assertEqual(conv_out.shape, [1, 8, 14, 12])
        self.assertEqual(out.shape, [1, 112, 12])

    def test_argmax_op_transposer_keep_dims(self):
        conv = paddle.nn.Conv2D(3, 8, (3, 3))
        data = paddle.rand([1, 3, 16, 14])
        with paddle.amp.auto_cast(level="O2"):
            conv_out = conv(data)
            # conv_out.shape = [1, 14, 12, 8] with NHWC
            out = paddle.argmax(conv_out, axis=1, keepdim=True)
        self.assertEqual(conv_out.shape, [1, 8, 14, 12])
        self.assertEqual(out.shape, [1, 1, 14, 12])

    def test_concat_op_transposer(self):
        in1 = paddle.rand([1, 8, 14, 12])
        conv = paddle.nn.Conv2D(3, 8, (3, 3))
        data = paddle.rand([1, 3, 16, 14])
        with paddle.amp.auto_cast(level="O2"):
            conv_out = conv(data)
            # conv_out.shape = [1, 14, 12, 8] with NHWC
            out = paddle.concat(x=[conv_out, in1], axis=0)

        self.assertEqual(conv_out.shape, [1, 8, 14, 12])
        self.assertEqual(out.shape, [2, 8, 14, 12])


if __name__ == '__main__':
    unittest.main()
