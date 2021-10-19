#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import random
import sys
import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers

paddle.enable_static()


class TestResNetUnitOp(unittest.TestCase):
    '''
    resnet_unit op is a fused op, and only has cuda kernel for float16.
    '''

    def config(self, fuse_add=False, has_shortcut=False):
        self.dtype = "float16"
        self.param_dtype = "float32"

        self.x_shape = [1, 56, 56, 64]
        self.x_np = np.random.random(self.x_shape).astype(self.dtype)

        self.num_channels_x = 64
        self.filter_size_x = 1
        self.num_filters = 256
        self.stride_x = 1

        self.w_x_shape = [256, 64, 1, 1]
        self.bn_scale_x_shape = [256]
        self.bn_bias_x_shape = [256]
        self.w_x_np = np.random.random(self.w_x_shape).astype(self.dtype)
        self.bn_scale_x_np = np.random.random(self.bn_scale_x_shape).astype(
            self.param_dtype)
        self.bn_bias_x_np = np.random.random(self.bn_bias_x_shape).astype(
            self.param_dtype)

        if has_shortcut:
            self.z_dtype = "float16"
            self.z_shape = [1, 56, 56, 64]
            self.z_np = np.random.random(self.z_shape).astype(self.dtype)

            self.num_channels_z = 64
            self.filter_size_z = 1
            self.stride_z = 1

            self.w_z_shape = [256, 64, 1, 1]
            self.bn_scale_z_shape = [256]
            self.bn_bias_z_shape = [256]
            self.w_z_np = np.random.random(self.w_z_shape).astype(self.dtype)
            self.bn_scale_z_np = np.random.random(self.bn_scale_z_shape).astype(
                self.param_dtype)
            self.bn_bias_z_np = np.random.random(self.bn_bias_z_shape).astype(
                self.param_dtype)
        elif fuse_add:
            self.z_dtype = self.x_dtype
            self.z_shape = self.x_shape
            self.z_np = np.random.random(self.z_shape).astype(self.dtype)

    def _define_param_attr(self,
                           test_fused_op,
                           fuse_add=False,
                           has_shortcut=False):
        def _reshape(array):
            if test_fused_op and len(array.shape) == 1:
                return np.reshape(array, [1, 1, 1, array.shape[0]])
            else:
                return array

        def _transpose(array):
            if test_fused_op:
                return np.transpose(array, axes=[0, 2, 3, 1])
            else:
                return array

        self.conv_x_param_attr = fluid.ParamAttr(
            name='conv2d_x.w',
            initializer=fluid.initializer.NumpyArrayInitializer(
                value=_transpose(self.w_x_np)))
        self.bn_scale_x_attr = fluid.ParamAttr(
            name='bn_x.w',
            initializer=fluid.initializer.NumpyArrayInitializer(
                value=_reshape(self.bn_scale_x_np)))
        self.bn_bias_x_attr = fluid.ParamAttr(
            name='bn_x.b',
            initializer=fluid.initializer.NumpyArrayInitializer(
                value=_reshape(self.bn_bias_x_np)))
        if has_shortcut:
            self.conv_z_param_attr = fluid.ParamAttr(
                name='conv2d_z.w',
                initializer=fluid.initializer.NumpyArrayInitializer(
                    value=_transpose(self.w_z_np)))
            self.bn_scale_z_attr = fluid.ParamAttr(
                name='bn_z.w',
                initializer=fluid.initializer.NumpyArrayInitializer(
                    value=_reshape(self.bn_scale_z_np)))
            self.bn_bias_z_attr = fluid.ParamAttr(
                name='bn_z.b',
                initializer=fluid.initializer.NumpyArrayInitializer(
                    value=_reshape(self.bn_bias_z_np)))
        else:
            self.conv_z_param_attr = None
            self.bn_scale_z_attr = None
            self.bn_bias_z_attr = None

    def _build_fused_program(self, fuse_add=False, has_shortcut=False):
        self._define_param_attr(True, fuse_add, has_shortcut)
        x = paddle.static.data(name='x', shape=self.x_shape, dtype=self.dtype)
        resnet_unit = paddle.incubate.operators.ResNetUnit(
            num_channels_x=64,
            num_filters=256,
            filter_size=1,
            stride=1,
            fuse_add=fuse_add,
            has_shortcut=has_shortcut,
            filter_x_attr=self.conv_x_param_attr,
            scale_x_attr=self.bn_scale_x_attr,
            bias_x_attr=self.bn_bias_x_attr,
            num_channels_z=64,
            filter_z_attr=self.conv_z_param_attr,
            scale_z_attr=self.bn_scale_z_attr,
            bias_z_attr=self.bn_bias_z_attr)
        y = resnet_unit(x, x)
        return y

    def _build_origin_program(self, fuse_add=False, has_shortcut=False):
        self._define_param_attr(False, fuse_add, has_shortcut)
        x = paddle.static.data(name='x', shape=self.x_shape, dtype=self.dtype)
        conv_x = fluid.layers.conv2d(
            input=x,
            filter_size=self.filter_size_x,
            num_filters=self.num_filters,
            stride=self.stride_x,
            padding=0,
            act=None,
            param_attr=self.conv_x_param_attr,
            bias_attr=False,
            data_format='NHWC')
        bn_x = fluid.layers.batch_norm(
            input=conv_x,
            param_attr=self.bn_scale_x_attr,
            bias_attr=self.bn_bias_x_attr,
            act=None,
            data_layout='NHWC')
        if has_shortcut:
            z = paddle.static.data(
                name='z', shape=self.z_shape, dtype=self.dtype)
            conv_z = fluid.layers.conv2d(
                input=z,
                filter_size=self.filter_size_z,
                num_filters=self.num_filters,
                stride=self.stride_z,
                padding=0,
                act=None,
                param_attr=self.conv_z_param_attr,
                bias_attr=False,
                data_format='NHWC')
            bn_z = fluid.layers.batch_norm(
                input=conv_z,
                param_attr=self.bn_scale_z_attr,
                bias_attr=self.bn_bias_z_attr,
                act=None,
                data_layout='NHWC')
            y = bn_x + bn_z
        elif fuse_add:
            z = paddle.static.data(
                name='z', shape=self.z_shape, dtype=self.dtype)
            y = bn_x + z
        else:
            y = bn_x
        y = fluid.layers.relu(y)
        return y

    def _run_program(self, exe, feed_dict, fetch_list):
        with fluid.scope_guard(fluid.Scope()):
            exe.run(fluid.default_startup_program())
            fetches = exe.run(program=fluid.default_main_program(),
                              feed=feed_dict,
                              fetch_list=fetch_list)
        return fetches

    def check(self, exe, feed_dict, fuse_add=False, has_shortcut=False):
        paddle.set_default_dtype(self.dtype)

        # build_fused_program: turn on resnet_unit_op
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            y = self._build_fused_program(
                fuse_add=fuse_add, has_shortcut=has_shortcut)
            y_fused = self._run_program(
                exe, feed_dict=feed_dict, fetch_list=[y])

        # build_origin_program: turn off resnet_unit_op
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            y = self._build_origin_program(
                fuse_add=fuse_add, has_shortcut=has_shortcut)
            y_origin = self._run_program(
                exe, feed_dict=feed_dict, fetch_list=[y])

        def _assert_is_close(a, b, atol=1e-2, rtol=1e-2):
            a = a.astype("float32") if a.dtype == np.float16 else a
            b = b.astype("float32") if b.dtype == np.float16 else b
            abs_a = np.abs(a)
            abs_a[abs_a < 1e-7] = 1e-3
            diff_mat = np.abs(a - b) / abs_a
            offset = np.argmax(diff_mat > rtol)
            self.assertTrue(
                np.allclose(
                    a, b, atol=atol, rtol=rtol),
                "The %d-th result has diff: %.5f vs %.5f" %
                (offset, a.flatten()[offset], b.flatten()[offset]))

        _assert_is_close(y_origin[0], y_fused[0])

    def test_resnet_unit(self):
        exe = fluid.Executor(fluid.CUDAPlace(0))
        self.config(fuse_add=False, has_shortcut=False)
        feed_dict = {"x": self.x_np}
        self.check(exe, feed_dict, fuse_add=False, has_shortcut=False)


if __name__ == '__main__':
    unittest.main()
