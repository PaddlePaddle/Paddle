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

import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle import base
from paddle.base import core
from paddle.base.framework import Program, program_guard
from paddle.base.layer_helper import LayerHelper


def transpose_layout(x, src_layout, dst_layout):
    return x.transpose([0, 2, 3, 1])


# default kNCHW
class TestTransferLayoutOpkNCHWTokNHWC(OpTest):
    def setUp(self):
        ipt = np.random.random(size=[2, 3, 10, 10])
        self.inputs = {'X': ipt.astype('float32')}
        self.outputs = {'Out': ipt.transpose([0, 2, 3, 1])}
        self.attrs = {'src_layout': 0, 'dst_layout': 1}  # kNHWC
        self.python_api = transpose_layout
        self.op_type = 'transfer_layout'

    def test_check_output(self):
        self.check_output()


def softmax_with_data_format(x, data_format, axis=-1, dtype=None, name=None):
    helper = LayerHelper("softmax", **locals())
    outs_cast = x

    outs_softmax = helper.create_variable_for_type_inference(outs_cast.dtype)
    helper.append_op(
        type='softmax',
        inputs={'X': outs_cast},
        outputs={'Out': outs_softmax},
        attrs={'axis': axis, 'use_cudnn': True, 'data_format': data_format},
    )

    return outs_softmax


class TestTransferLayoutOpGpu(unittest.TestCase):
    def test_layout_transfer(self):
        with paddle.pir_utils.OldIrGuard():
            if not core.is_compiled_with_cuda():
                return

            paddle.enable_static()

            main_program = Program()
            startup_program = Program()
            n, c, h, w = 2, 3, 4, 5
            with program_guard(main_program, startup_program):
                x = paddle.static.data(
                    shape=[n, c, h, w], dtype='float32', name='x'
                )
                y = softmax_with_data_format(x, data_format='NCHW')
                z = softmax_with_data_format(y, data_format='NHWC')

            place = (
                base.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else base.CPUPlace()
            )
            exe = base.Executor(place)
            exe.run(startup_program)
            ret = exe.run(
                main_program,
                feed={'x': np.full((n, c, h, w), 1, np.float32)},
                fetch_list=[z.name],
            )
            assert len(ret) == 1
            assert ret[0].shape == (n, h, w, c)


class TestTransferLayoutFP16Op(OpTest):
    def setUp(self):
        self.op_type = 'transfer_layout'
        self.dtype = np.float16
        x = np.random.random(size=[2, 5, 10, 10])
        self.inputs = {'X': x.astype(self.dtype)}
        self.outputs = {'Out': x.transpose([0, 2, 3, 1])}
        self.attrs = {'src_layout': 0, 'dst_layout': 1}
        self.python_api = transpose_layout

    def test_check_output(self):
        self.check_output()


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestTransferLayoutBP16Op(OpTest):
    def setUp(self):
        self.op_type = 'transfer_layout'
        self.dtype = np.uint16
        x = np.random.random(size=[2, 5, 10, 10])
        self.inputs = {'X': convert_float_to_uint16(x.astype('float32'))}
        self.outputs = {
            'Out': convert_float_to_uint16(
                x.transpose([0, 2, 3, 1]), data_format="NHWC"
            )
        }
        self.attrs = {'src_layout': 0, 'dst_layout': 1}
        self.python_api = transpose_layout

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
