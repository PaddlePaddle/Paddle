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

# repo: PaddleDetection
# model: configs^ssd^ssd_vgg16_300_240e_voc_single_dy2st_train
# api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.pooling.max_pool2d,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.pooling.max_pool2d,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.pooling.max_pool2d,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.pooling.max_pool2d,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.pooling.max_pool2d,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.norm.normalize,method||unsqueeze,method||unsqueeze,method||unsqueeze,method||__mul__
import unittest

import numpy as np

import paddle


class SIR2(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_368 = self.create_parameter(
            shape=[256, 128, 3, 3],
            dtype=paddle.float32,
        )
        self.var_350 = self.create_parameter(
            shape=[64, 3, 3, 3],
            dtype=paddle.float32,
        )
        self.var_427 = self.create_parameter(
            shape=[256, 128, 3, 3],
            dtype=paddle.float32,
        )
        self.var_351 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.var_440 = self.create_parameter(
            shape=[128],
            dtype=paddle.float32,
        )
        self.var_424 = self.create_parameter(
            shape=[128],
            dtype=paddle.float32,
        )
        self.var_369 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_390 = self.create_parameter(
            shape=[512],
            dtype=paddle.float32,
        )
        self.var_354 = self.create_parameter(
            shape=[64, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.var_439 = self.create_parameter(
            shape=[128, 256, 1, 1],
            dtype=paddle.float32,
        )
        self.var_381 = self.create_parameter(
            shape=[512, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_386 = self.create_parameter(
            shape=[512],
            dtype=paddle.float32,
        )
        self.var_373 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_399 = self.create_parameter(
            shape=[512],
            dtype=paddle.float32,
        )
        self.var_363 = self.create_parameter(
            shape=[128, 128, 3, 3],
            dtype=paddle.float32,
        )
        self.var_411 = self.create_parameter(
            shape=[1024, 1024, 1, 1],
            dtype=paddle.float32,
        )
        self.var_377 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_428 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_432 = self.create_parameter(
            shape=[128],
            dtype=paddle.float32,
        )
        self.var_394 = self.create_parameter(
            shape=[512, 512, 3, 3],
            dtype=paddle.float32,
        )
        self.var_372 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_385 = self.create_parameter(
            shape=[512, 512, 3, 3],
            dtype=paddle.float32,
        )
        self.var_412 = self.create_parameter(
            shape=[1024],
            dtype=paddle.float32,
        )
        self.var_407 = self.create_parameter(
            shape=[1024, 512, 3, 3],
            dtype=paddle.float32,
        )
        self.var_395 = self.create_parameter(
            shape=[512],
            dtype=paddle.float32,
        )
        self.var_389 = self.create_parameter(
            shape=[512, 512, 3, 3],
            dtype=paddle.float32,
        )
        self.var_431 = self.create_parameter(
            shape=[128, 256, 1, 1],
            dtype=paddle.float32,
        )
        self.var_435 = self.create_parameter(
            shape=[256, 128, 3, 3],
            dtype=paddle.float32,
        )
        self.var_408 = self.create_parameter(
            shape=[1024],
            dtype=paddle.float32,
        )
        self.var_398 = self.create_parameter(
            shape=[512, 512, 3, 3],
            dtype=paddle.float32,
        )
        self.var_376 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_415 = self.create_parameter(
            shape=[256, 1024, 1, 1],
            dtype=paddle.float32,
        )
        self.var_444 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_420 = self.create_parameter(
            shape=[512],
            dtype=paddle.float32,
        )
        self.var_355 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.var_382 = self.create_parameter(
            shape=[512],
            dtype=paddle.float32,
        )
        self.var_364 = self.create_parameter(
            shape=[128],
            dtype=paddle.float32,
        )
        self.var_448 = self.create_parameter(
            shape=[512],
            dtype=paddle.float32,
        )
        self.var_359 = self.create_parameter(
            shape=[128, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.var_360 = self.create_parameter(
            shape=[128],
            dtype=paddle.float32,
        )
        self.var_402 = self.create_parameter(
            shape=[512, 512, 3, 3],
            dtype=paddle.float32,
        )
        self.var_416 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_423 = self.create_parameter(
            shape=[128, 512, 1, 1],
            dtype=paddle.float32,
        )
        self.var_419 = self.create_parameter(
            shape=[512, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_403 = self.create_parameter(
            shape=[512],
            dtype=paddle.float32,
        )
        self.var_436 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_443 = self.create_parameter(
            shape=[256, 128, 3, 3],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_347,  # (shape: [1, 3, 300, 300], dtype: paddle.float32, stop_gradient: True)
    ):
        var_352 = paddle.nn.functional.conv._conv_nd(
            var_347,
            self.var_350,
            bias=self.var_351,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_353 = paddle.nn.functional.activation.relu(var_352)
        var_356 = paddle.nn.functional.conv._conv_nd(
            var_353,
            self.var_354,
            bias=self.var_355,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_357 = paddle.nn.functional.activation.relu(var_356)
        var_358 = paddle.nn.functional.pooling.max_pool2d(
            var_357,
            kernel_size=2,
            stride=2,
            padding=0,
            return_mask=False,
            ceil_mode=True,
            data_format='NCHW',
            name=None,
        )
        var_361 = paddle.nn.functional.conv._conv_nd(
            var_358,
            self.var_359,
            bias=self.var_360,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_362 = paddle.nn.functional.activation.relu(var_361)
        var_365 = paddle.nn.functional.conv._conv_nd(
            var_362,
            self.var_363,
            bias=self.var_364,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_366 = paddle.nn.functional.activation.relu(var_365)
        var_367 = paddle.nn.functional.pooling.max_pool2d(
            var_366,
            kernel_size=2,
            stride=2,
            padding=0,
            return_mask=False,
            ceil_mode=True,
            data_format='NCHW',
            name=None,
        )
        var_370 = paddle.nn.functional.conv._conv_nd(
            var_367,
            self.var_368,
            bias=self.var_369,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_371 = paddle.nn.functional.activation.relu(var_370)
        var_374 = paddle.nn.functional.conv._conv_nd(
            var_371,
            self.var_372,
            bias=self.var_373,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_375 = paddle.nn.functional.activation.relu(var_374)
        var_378 = paddle.nn.functional.conv._conv_nd(
            var_375,
            self.var_376,
            bias=self.var_377,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_379 = paddle.nn.functional.activation.relu(var_378)
        var_380 = paddle.nn.functional.pooling.max_pool2d(
            var_379,
            kernel_size=2,
            stride=2,
            padding=0,
            return_mask=False,
            ceil_mode=True,
            data_format='NCHW',
            name=None,
        )
        var_383 = paddle.nn.functional.conv._conv_nd(
            var_380,
            self.var_381,
            bias=self.var_382,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_384 = paddle.nn.functional.activation.relu(var_383)
        var_387 = paddle.nn.functional.conv._conv_nd(
            var_384,
            self.var_385,
            bias=self.var_386,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_388 = paddle.nn.functional.activation.relu(var_387)
        var_391 = paddle.nn.functional.conv._conv_nd(
            var_388,
            self.var_389,
            bias=self.var_390,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_392 = paddle.nn.functional.activation.relu(var_391)
        var_393 = paddle.nn.functional.pooling.max_pool2d(
            var_392,
            kernel_size=2,
            stride=2,
            padding=0,
            return_mask=False,
            ceil_mode=True,
            data_format='NCHW',
            name=None,
        )
        var_396 = paddle.nn.functional.conv._conv_nd(
            var_393,
            self.var_394,
            bias=self.var_395,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_397 = paddle.nn.functional.activation.relu(var_396)
        var_400 = paddle.nn.functional.conv._conv_nd(
            var_397,
            self.var_398,
            bias=self.var_399,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_401 = paddle.nn.functional.activation.relu(var_400)
        var_404 = paddle.nn.functional.conv._conv_nd(
            var_401,
            self.var_402,
            bias=self.var_403,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_405 = paddle.nn.functional.activation.relu(var_404)
        var_406 = paddle.nn.functional.pooling.max_pool2d(
            var_405,
            kernel_size=3,
            stride=1,
            padding=1,
            return_mask=False,
            ceil_mode=True,
            data_format='NCHW',
            name=None,
        )
        var_409 = paddle.nn.functional.conv._conv_nd(
            var_406,
            self.var_407,
            bias=self.var_408,
            stride=[1, 1],
            padding=[6, 6],
            padding_algorithm='EXPLICIT',
            dilation=[6, 6],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_410 = paddle.nn.functional.activation.relu(var_409)
        var_413 = paddle.nn.functional.conv._conv_nd(
            var_410,
            self.var_411,
            bias=self.var_412,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_414 = paddle.nn.functional.activation.relu(var_413)
        var_417 = paddle.nn.functional.conv._conv_nd(
            var_414,
            self.var_415,
            bias=self.var_416,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_418 = paddle.nn.functional.activation.relu(var_417)
        var_421 = paddle.nn.functional.conv._conv_nd(
            var_418,
            self.var_419,
            bias=self.var_420,
            stride=[2, 2],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_422 = paddle.nn.functional.activation.relu(var_421)
        var_425 = paddle.nn.functional.conv._conv_nd(
            var_422,
            self.var_423,
            bias=self.var_424,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_426 = paddle.nn.functional.activation.relu(var_425)
        var_429 = paddle.nn.functional.conv._conv_nd(
            var_426,
            self.var_427,
            bias=self.var_428,
            stride=[2, 2],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_430 = paddle.nn.functional.activation.relu(var_429)
        var_433 = paddle.nn.functional.conv._conv_nd(
            var_430,
            self.var_431,
            bias=self.var_432,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_434 = paddle.nn.functional.activation.relu(var_433)
        var_437 = paddle.nn.functional.conv._conv_nd(
            var_434,
            self.var_435,
            bias=self.var_436,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_438 = paddle.nn.functional.activation.relu(var_437)
        var_441 = paddle.nn.functional.conv._conv_nd(
            var_438,
            self.var_439,
            bias=self.var_440,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_442 = paddle.nn.functional.activation.relu(var_441)
        var_445 = paddle.nn.functional.conv._conv_nd(
            var_442,
            self.var_443,
            bias=self.var_444,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_446 = paddle.nn.functional.activation.relu(var_445)
        var_447 = paddle.nn.functional.norm.normalize(
            var_392, axis=1, epsilon=1e-10
        )
        var_449 = self.var_448.unsqueeze(0)
        var_450 = var_449.unsqueeze(2)
        var_451 = var_450.unsqueeze(3)
        var_452 = var_451.__mul__(var_447)
        return var_452, var_414, var_422, var_430, var_438, var_446


class TestSIR2(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 3, 300, 300], dtype=paddle.float32),
        )
        self.net = SIR2()

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        paddle.set_flags({'FLAGS_prim_all': with_prim})
        if to_static:
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(
                    net, build_strategy=build_strategy, full_graph=True
                )
            else:
                net = paddle.jit.to_static(net, full_graph=True)
        outs = net(*self.inputs)
        return outs

    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(
            self.net, to_static=True, with_prim=True, with_cinn=True
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
