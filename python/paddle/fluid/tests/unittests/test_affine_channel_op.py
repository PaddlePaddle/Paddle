#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
"""
Unit testing for affine_channel_op
"""

import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid.core as core
import paddle.fluid as fluid


def affine_channel(x, scale, bias, layout):
    C = x.shape[1] if layout == 'NCHW' else x.shape[-1]
    if len(x.shape) == 4:
        new_shape = (1, C, 1, 1) if layout == 'NCHW' else (1, 1, 1, C)
    else:
        new_shape = (1, C)
    scale = scale.reshape(new_shape)
    bias = bias.reshape(new_shape)
    return x * scale + bias


class TestAffineChannelOp(OpTest):

    def setUp(self):
        self.op_type = "affine_channel"
        self.init_test_case()

        x = np.random.random(self.shape).astype("float64")
        scale = np.random.random(self.C).astype("float64")
        bias = np.random.random(self.C).astype("float64")

        y = affine_channel(x, scale, bias, self.layout)

        self.inputs = {'X': x, 'Scale': scale, 'Bias': bias}
        self.attrs = {'data_layout': self.layout}
        self.outputs = {'Out': y}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'Scale', 'Bias'], 'Out')

    def test_check_grad_stopgrad_dx(self):
        self.check_grad(['Scale', 'Bias'], 'Out', no_grad_set=set('X'))

    def test_check_grad_stopgrad_dscale_dbias(self):
        self.check_grad(['X'], 'Out', no_grad_set=set(['Scale', 'Bias']))

    def init_test_case(self):
        self.shape = [2, 100, 3, 3]
        self.C = 100
        self.layout = 'NCHW'


class TestAffineChannelOpError(unittest.TestCase):

    def test_errors(self):
        with fluid.program_guard(fluid.Program()):

            def test_x_type():
                input_data = np.random.random(2, 1, 2, 2).astype("float32")
                fluid.layers.affine_channel(input_data)

            self.assertRaises(TypeError, test_x_type)

            def test_x_dtype():
                x2 = fluid.layers.data(name='x2',
                                       shape=[None, 1, 2, 2],
                                       dtype='int32')
                fluid.layers.affine_channel(x2)

            self.assertRaises(TypeError, test_x_dtype)

            def test_scale_type():
                x3 = fluid.layers.data(name='x3',
                                       shape=[None, 1, 2, 2],
                                       dtype='float32')
                fluid.layers.affine_channel(x3, scale=1)

            self.assertRaises(TypeError, test_scale_type)

            def test_bias_type():
                x4 = fluid.layers.data(name='x4',
                                       shape=[None, 1, 2, 2],
                                       dtype='float32')
                fluid.layers.affine_channel(x4, bias=1)

            self.assertRaises(TypeError, test_bias_type)


class TestAffineChannelNHWC(TestAffineChannelOp):

    def init_test_case(self):
        self.shape = [2, 3, 3, 100]
        self.C = 100
        self.layout = 'NHWC'

    def test_check_grad_stopgrad_dx(self):
        return

    def test_check_grad_stopgrad_dscale_dbias(self):
        return


class TestAffineChannel2D(TestAffineChannelOp):

    def init_test_case(self):
        self.shape = [2, 100]
        self.C = 100
        self.layout = 'NCHW'

    def test_check_grad_stopgrad_dx(self):
        return

    def test_check_grad_stopgrad_dscale_dbias(self):
        return


# TODO(qingqing): disable unit testing for large shape
#class TestAffineChannelNCHWLargeShape(TestAffineChannelOp):
#    def init_test_case(self):
#        self.shape = [4, 128, 112, 112]
#        self.C = 128
#        self.layout = 'NCHW'
#
#    # since the gradient check is very slow in large shape, so skip check_grad
#    def test_check_grad(self):
#        pass
#
#    def test_check_grad_stopgrad_dx(self):
#        pass
#
#    def test_check_grad_stopgrad_dscale_dbias(self):
#        pass

#class TestAffineChannelNHWCLargeShape(TestAffineChannelNCHWLargeShape):
#    def init_test_case(self):
#        self.shape = [64, 32, 32, 128]
#        self.C = 128
#        self.layout = 'NHWC'

if __name__ == '__main__':
    unittest.main()
