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

# repo: PaddleClas
# model: ppcls^configs^ImageNet^SqueezeNet^SqueezeNet1_0
# api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.pooling.max_pool2d,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.tensor.manipulation.concat,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.tensor.manipulation.concat,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.tensor.manipulation.concat,api||paddle.nn.functional.pooling.max_pool2d,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.tensor.manipulation.concat,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.tensor.manipulation.concat,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.tensor.manipulation.concat,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.tensor.manipulation.concat,api||paddle.nn.functional.pooling.max_pool2d,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.tensor.manipulation.concat,api||paddle.nn.functional.common.dropout,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.pooling.adaptive_avg_pool2d,api||paddle.tensor.manipulation.squeeze
import unittest

import numpy as np

import paddle


class SIR3(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_389 = self.create_parameter(
            shape=[32, 128, 1, 1],
            dtype=paddle.float32,
        )
        self.var_364 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.var_367 = self.create_parameter(
            shape=[64, 16, 1, 1],
            dtype=paddle.float32,
        )
        self.var_397 = self.create_parameter(
            shape=[128, 32, 3, 3],
            dtype=paddle.float32,
        )
        self.var_376 = self.create_parameter(
            shape=[16, 128, 1, 1],
            dtype=paddle.float32,
        )
        self.var_420 = self.create_parameter(
            shape=[192, 48, 1, 1],
            dtype=paddle.float32,
        )
        self.var_411 = self.create_parameter(
            shape=[128, 32, 3, 3],
            dtype=paddle.float32,
        )
        self.var_380 = self.create_parameter(
            shape=[64, 16, 1, 1],
            dtype=paddle.float32,
        )
        self.var_421 = self.create_parameter(
            shape=[192],
            dtype=paddle.float32,
        )
        self.var_461 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_384 = self.create_parameter(
            shape=[64, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.var_358 = self.create_parameter(
            shape=[96, 3, 7, 7],
            dtype=paddle.float32,
        )
        self.var_416 = self.create_parameter(
            shape=[48, 256, 1, 1],
            dtype=paddle.float32,
        )
        self.var_442 = self.create_parameter(
            shape=[64, 384, 1, 1],
            dtype=paddle.float32,
        )
        self.var_403 = self.create_parameter(
            shape=[32, 256, 1, 1],
            dtype=paddle.float32,
        )
        self.var_377 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.var_412 = self.create_parameter(
            shape=[128],
            dtype=paddle.float32,
        )
        self.var_363 = self.create_parameter(
            shape=[16, 96, 1, 1],
            dtype=paddle.float32,
        )
        self.var_417 = self.create_parameter(
            shape=[48],
            dtype=paddle.float32,
        )
        self.var_424 = self.create_parameter(
            shape=[192, 48, 3, 3],
            dtype=paddle.float32,
        )
        self.var_443 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.var_450 = self.create_parameter(
            shape=[256, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.var_456 = self.create_parameter(
            shape=[64, 512, 1, 1],
            dtype=paddle.float32,
        )
        self.var_381 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.var_433 = self.create_parameter(
            shape=[192, 48, 1, 1],
            dtype=paddle.float32,
        )
        self.var_407 = self.create_parameter(
            shape=[128, 32, 1, 1],
            dtype=paddle.float32,
        )
        self.var_393 = self.create_parameter(
            shape=[128, 32, 1, 1],
            dtype=paddle.float32,
        )
        self.var_385 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.var_368 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.var_425 = self.create_parameter(
            shape=[192],
            dtype=paddle.float32,
        )
        self.var_446 = self.create_parameter(
            shape=[256, 64, 1, 1],
            dtype=paddle.float32,
        )
        self.var_447 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_390 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.var_359 = self.create_parameter(
            shape=[96],
            dtype=paddle.float32,
        )
        self.var_429 = self.create_parameter(
            shape=[48, 384, 1, 1],
            dtype=paddle.float32,
        )
        self.var_451 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_471 = self.create_parameter(
            shape=[1000],
            dtype=paddle.float32,
        )
        self.var_470 = self.create_parameter(
            shape=[1000, 512, 1, 1],
            dtype=paddle.float32,
        )
        self.var_460 = self.create_parameter(
            shape=[256, 64, 1, 1],
            dtype=paddle.float32,
        )
        self.var_408 = self.create_parameter(
            shape=[128],
            dtype=paddle.float32,
        )
        self.var_404 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.var_398 = self.create_parameter(
            shape=[128],
            dtype=paddle.float32,
        )
        self.var_430 = self.create_parameter(
            shape=[48],
            dtype=paddle.float32,
        )
        self.var_464 = self.create_parameter(
            shape=[256, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.var_371 = self.create_parameter(
            shape=[64, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.var_438 = self.create_parameter(
            shape=[192],
            dtype=paddle.float32,
        )
        self.var_457 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.var_372 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.var_434 = self.create_parameter(
            shape=[192],
            dtype=paddle.float32,
        )
        self.var_437 = self.create_parameter(
            shape=[192, 48, 3, 3],
            dtype=paddle.float32,
        )
        self.var_394 = self.create_parameter(
            shape=[128],
            dtype=paddle.float32,
        )
        self.var_465 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_357,  # (shape: [10, 3, 224, 224], dtype: paddle.float32, stop_gradient: True)
    ):
        var_360 = paddle.nn.functional.conv._conv_nd(
            var_357,
            self.var_358,
            bias=self.var_359,
            stride=[2, 2],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_361 = paddle.nn.functional.activation.relu(var_360)
        var_362 = paddle.nn.functional.pooling.max_pool2d(
            var_361,
            kernel_size=3,
            stride=2,
            padding=0,
            return_mask=False,
            ceil_mode=False,
            data_format='NCHW',
            name=None,
        )
        var_365 = paddle.nn.functional.conv._conv_nd(
            var_362,
            self.var_363,
            bias=self.var_364,
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
        var_366 = paddle.nn.functional.activation.relu(var_365)
        var_369 = paddle.nn.functional.conv._conv_nd(
            var_366,
            self.var_367,
            bias=self.var_368,
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
        var_370 = paddle.nn.functional.activation.relu(var_369)
        var_373 = paddle.nn.functional.conv._conv_nd(
            var_366,
            self.var_371,
            bias=self.var_372,
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
        var_374 = paddle.nn.functional.activation.relu(var_373)
        var_375 = paddle.tensor.manipulation.concat([var_370, var_374], axis=1)
        var_378 = paddle.nn.functional.conv._conv_nd(
            var_375,
            self.var_376,
            bias=self.var_377,
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
        var_379 = paddle.nn.functional.activation.relu(var_378)
        var_382 = paddle.nn.functional.conv._conv_nd(
            var_379,
            self.var_380,
            bias=self.var_381,
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
        var_383 = paddle.nn.functional.activation.relu(var_382)
        var_386 = paddle.nn.functional.conv._conv_nd(
            var_379,
            self.var_384,
            bias=self.var_385,
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
        var_387 = paddle.nn.functional.activation.relu(var_386)
        var_388 = paddle.tensor.manipulation.concat([var_383, var_387], axis=1)
        var_391 = paddle.nn.functional.conv._conv_nd(
            var_388,
            self.var_389,
            bias=self.var_390,
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
        var_392 = paddle.nn.functional.activation.relu(var_391)
        var_395 = paddle.nn.functional.conv._conv_nd(
            var_392,
            self.var_393,
            bias=self.var_394,
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
        var_396 = paddle.nn.functional.activation.relu(var_395)
        var_399 = paddle.nn.functional.conv._conv_nd(
            var_392,
            self.var_397,
            bias=self.var_398,
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
        var_400 = paddle.nn.functional.activation.relu(var_399)
        var_401 = paddle.tensor.manipulation.concat([var_396, var_400], axis=1)
        var_402 = paddle.nn.functional.pooling.max_pool2d(
            var_401,
            kernel_size=3,
            stride=2,
            padding=0,
            return_mask=False,
            ceil_mode=False,
            data_format='NCHW',
            name=None,
        )
        var_405 = paddle.nn.functional.conv._conv_nd(
            var_402,
            self.var_403,
            bias=self.var_404,
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
        var_406 = paddle.nn.functional.activation.relu(var_405)
        var_409 = paddle.nn.functional.conv._conv_nd(
            var_406,
            self.var_407,
            bias=self.var_408,
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
        var_410 = paddle.nn.functional.activation.relu(var_409)
        var_413 = paddle.nn.functional.conv._conv_nd(
            var_406,
            self.var_411,
            bias=self.var_412,
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
        var_414 = paddle.nn.functional.activation.relu(var_413)
        var_415 = paddle.tensor.manipulation.concat([var_410, var_414], axis=1)
        var_418 = paddle.nn.functional.conv._conv_nd(
            var_415,
            self.var_416,
            bias=self.var_417,
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
        var_419 = paddle.nn.functional.activation.relu(var_418)
        var_422 = paddle.nn.functional.conv._conv_nd(
            var_419,
            self.var_420,
            bias=self.var_421,
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
        var_423 = paddle.nn.functional.activation.relu(var_422)
        var_426 = paddle.nn.functional.conv._conv_nd(
            var_419,
            self.var_424,
            bias=self.var_425,
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
        var_427 = paddle.nn.functional.activation.relu(var_426)
        var_428 = paddle.tensor.manipulation.concat([var_423, var_427], axis=1)
        var_431 = paddle.nn.functional.conv._conv_nd(
            var_428,
            self.var_429,
            bias=self.var_430,
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
        var_432 = paddle.nn.functional.activation.relu(var_431)
        var_435 = paddle.nn.functional.conv._conv_nd(
            var_432,
            self.var_433,
            bias=self.var_434,
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
        var_436 = paddle.nn.functional.activation.relu(var_435)
        var_439 = paddle.nn.functional.conv._conv_nd(
            var_432,
            self.var_437,
            bias=self.var_438,
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
        var_440 = paddle.nn.functional.activation.relu(var_439)
        var_441 = paddle.tensor.manipulation.concat([var_436, var_440], axis=1)
        var_444 = paddle.nn.functional.conv._conv_nd(
            var_441,
            self.var_442,
            bias=self.var_443,
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
        var_445 = paddle.nn.functional.activation.relu(var_444)
        var_448 = paddle.nn.functional.conv._conv_nd(
            var_445,
            self.var_446,
            bias=self.var_447,
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
        var_449 = paddle.nn.functional.activation.relu(var_448)
        var_452 = paddle.nn.functional.conv._conv_nd(
            var_445,
            self.var_450,
            bias=self.var_451,
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
        var_453 = paddle.nn.functional.activation.relu(var_452)
        var_454 = paddle.tensor.manipulation.concat([var_449, var_453], axis=1)
        var_455 = paddle.nn.functional.pooling.max_pool2d(
            var_454,
            kernel_size=3,
            stride=2,
            padding=0,
            return_mask=False,
            ceil_mode=False,
            data_format='NCHW',
            name=None,
        )
        var_458 = paddle.nn.functional.conv._conv_nd(
            var_455,
            self.var_456,
            bias=self.var_457,
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
        var_459 = paddle.nn.functional.activation.relu(var_458)
        var_462 = paddle.nn.functional.conv._conv_nd(
            var_459,
            self.var_460,
            bias=self.var_461,
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
        var_463 = paddle.nn.functional.activation.relu(var_462)
        var_466 = paddle.nn.functional.conv._conv_nd(
            var_459,
            self.var_464,
            bias=self.var_465,
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
        var_467 = paddle.nn.functional.activation.relu(var_466)
        var_468 = paddle.tensor.manipulation.concat([var_463, var_467], axis=1)
        var_469 = paddle.nn.functional.common.dropout(
            var_468,
            p=0.5,
            axis=None,
            training=False,
            mode='downscale_in_infer',
            name=None,
        )
        var_472 = paddle.nn.functional.conv._conv_nd(
            var_469,
            self.var_470,
            bias=self.var_471,
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
        var_473 = paddle.nn.functional.activation.relu(var_472)
        var_474 = paddle.nn.functional.pooling.adaptive_avg_pool2d(
            var_473, output_size=1, data_format='NCHW', name=None
        )
        var_475 = paddle.tensor.manipulation.squeeze(var_474, axis=[2, 3])
        return var_475


class TestSIR3(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[10, 3, 224, 224], dtype=paddle.float32),
        )
        self.net = SIR3()

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
