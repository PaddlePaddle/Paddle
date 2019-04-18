# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest

import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid.proto.framework_pb2 as framework_pb2


class Relu2Test(unittest.TestCase):
    def initMethod(self):
        self.dtype = 'float32'
        self.shape = [110, 300]

    def cal_mask(self, x):
        x = list(x.flatten() >= 0)
        if len(x) % 8 != 0:
            x = x + [0] * (8 - len(x) % 8)
        x_reshaped = np.reshape(x, newshape=[8, -1])
        mask_bits = np.flip(x, axis=0)
        return np.flip(np.packbits(mask_bits), axis=0)

    def main(self, place):
        self.initMethod()
        x = (np.random.random(size=self.shape) - 0.5).astype(self.dtype)

        mask = self.cal_mask(x)
        out = np.maximum(x, 0).astype(self.dtype)

        scope = fluid.Scope()
        x_t = scope.var('X').get_tensor()
        x_t.set(x, place)

        out_t = scope.var('Out').get_tensor()
        mask_t = scope.var('Mask').get_tensor()

        op = Operator('relu2', X='X', Out='Out', Mask='Mask')
        op.run(scope, place)

        out_t_cal = np.array(out_t)
        mask_t_cal = np.array(mask_t)

        self.assertTrue((out == out_t_cal).all())
        self.assertTrue((mask == mask_t_cal).all())

        grad_op_desc = framework_pb2.OpDesc()
        grad_op_desc.type = 'relu2_grad'
        ipt = grad_op_desc.inputs.add()
        ipt.parameter = 'Out@GRAD'
        ipt.arguments.extend(['Out@GRAD'])

        ipt = grad_op_desc.inputs.add()
        ipt.parameter = 'Mask'
        ipt.arguments.extend(['Mask'])

        opt = grad_op_desc.outputs.add()
        opt.parameter = 'X@GRAD'
        opt.arguments.extend(['X@GRAD'])

        # NOTE(zjl): op_callback is necessay for all op, so I add it here 
        op_callback = grad_op_desc.attrs.add()
        op_callback.name = core.op_proto_and_checker_maker.kOpCreationCallstackAttrName(
        )
        op_callback.type = framework_pb2.STRINGS
        op_callback.strings.extend(["op_callback ommitted"])

        x_grad_t = scope.var('X@GRAD').get_tensor()
        out_grad_t = scope.var('Out@GRAD').get_tensor()

        out_grad = (np.random.random(size=self.shape) - 0.5).astype(self.dtype)
        out_grad_t.set(out_grad, place)

        bwd_op = core.Operator.create(grad_op_desc.SerializeToString())
        bwd_op.run(scope, place)
        x_grad = np.array(x_grad_t)

        self.check_grad(x, x_grad, out_grad)

    def check_grad(self, x, x_grad, out_grad):
        # x_grad = out_grad, x >= 0
        # x_grad = 0, x < 0
        self.assertTrue(x.shape == x_grad.shape and x.shape == out_grad.shape)
        zero_idx1 = np.where((x_grad == out_grad).flatten())[0]
        zero_idx2 = np.where((x >= 0).flatten())[0]
        self.assertTrue((zero_idx1 == zero_idx2).all())

    def test_main(self):
        self.main(fluid.CPUPlace())
        if fluid.core.is_compiled_with_cuda():
            self.main(fluid.CUDAPlace(0))


class Relu2Test2(Relu2Test):
    def initMethod(self):
        self.dtype = 'float64'
        self.shape = [117, 315, 3]


if __name__ == '__main__':
    unittest.main()
