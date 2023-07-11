# Copyright (c) 2023 CINN Authors. All Rights Reserved.
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
from struct import pack, unpack

import numpy as np
from cinn.common import is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool

import paddle


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestBitcastConvertOp(OpTest):
    def setUp(self):
        self.init_case()

    # input[(3, 1), int32] --> output[(3, 1, 4), uint8]
    def init_case(self):
        data = np.random.random([3, 1]).astype(np.int32)
        packed = pack(data.size * 'i', *data.flatten())
        self.inputs = {"x": data}
        self.outputs = {
            "y": np.array(unpack('12B', packed), dtype='uint8').reshape(
                (3, 1, 4)
            ),
            "output_type": "uint8",
        }

    def build_paddle_program(self, target):
        y = paddle.to_tensor(self.outputs["y"], stop_gradient=False)
        self.paddle_outputs = [y]

    def build_cinn_program(self, target):
        builder = NetBuilder("bitcast_convert")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        out = builder.bitcast_convert(x, self.outputs["output_type"])
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])
        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestBitcastConvertCase1(TestBitcastConvertOp):
    # input[(4, 2), int16] --> output[(4), int32]
    def init_case(self):
        data = np.random.random([4, 2]).astype(np.int16)
        packed = pack(data.size * 'h', *data.flatten())
        self.inputs = {"x": data}
        self.outputs = {
            "y": np.array(unpack('4i', packed), dtype='int32').reshape(4),
            "output_type": "int32",
        }


class TestBitcastConvertCase2(TestBitcastConvertOp):
    # input[(4, 3, 2), float32] --> output[(4, 3), float64]
    def init_case(self):
        data = np.random.random([4, 3, 2]).astype(np.float32)
        packed = pack(data.size * 'f', *data.flatten())
        self.inputs = {"x": data}
        self.outputs = {
            "y": np.array(unpack('12d', packed), dtype='float64').reshape(
                (4, 3)
            ),
            "output_type": "float64",
        }


class TestBitcastConvertCase3(TestBitcastConvertOp):
    # input[(4, 3, 2), float32] --> output[(4, 3, 2, 2), uint16]
    def init_case(self):
        data = np.random.random([4, 3, 2]).astype(np.float32)
        packed = pack(data.size * 'f', *data.flatten())
        self.inputs = {"x": data}
        self.outputs = {
            "y": np.array(unpack('48H', packed), dtype='uint16').reshape(
                (4, 3, 2, 2)
            ),
            "output_type": "uint16",
        }


if __name__ == "__main__":
    unittest.main()
