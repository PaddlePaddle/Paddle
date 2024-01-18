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
# model: configs^retinanet^retinanet_r50_fpn_1x_coco_single_dy2st_train
# api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd
import unittest

import numpy as np

import paddle


class SIR34(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_382 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_389 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_385 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_370 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_386 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_397 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_378 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_404 = self.create_parameter(
            shape=[36, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_394 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_393 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_377 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_373 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_402 = self.create_parameter(
            shape=[720],
            dtype=paddle.float32,
        )
        self.var_369 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_398 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_405 = self.create_parameter(
            shape=[36],
            dtype=paddle.float32,
        )
        self.var_374 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_381 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_390 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_401 = self.create_parameter(
            shape=[720, 256, 3, 3],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_364,  # (shape: [1, 256, 84, 128], dtype: paddle.float32, stop_gradient: False)
        var_365,  # (shape: [1, 256, 42, 64], dtype: paddle.float32, stop_gradient: False)
        var_366,  # (shape: [1, 256, 21, 32], dtype: paddle.float32, stop_gradient: False)
        var_367,  # (shape: [1, 256, 11, 16], dtype: paddle.float32, stop_gradient: False)
        var_368,  # (shape: [1, 256, 6, 8], dtype: paddle.float32, stop_gradient: False)
    ):
        var_371 = paddle.nn.functional.conv._conv_nd(
            var_364,
            self.var_369,
            bias=self.var_370,
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
        var_372 = paddle.nn.functional.activation.relu(var_371)
        var_375 = paddle.nn.functional.conv._conv_nd(
            var_364,
            self.var_373,
            bias=self.var_374,
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
        var_376 = paddle.nn.functional.activation.relu(var_375)
        var_379 = paddle.nn.functional.conv._conv_nd(
            var_372,
            self.var_377,
            bias=self.var_378,
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
        var_380 = paddle.nn.functional.activation.relu(var_379)
        var_383 = paddle.nn.functional.conv._conv_nd(
            var_376,
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
            var_380,
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
            var_384,
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
        var_395 = paddle.nn.functional.conv._conv_nd(
            var_388,
            self.var_393,
            bias=self.var_394,
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
        var_403 = paddle.nn.functional.conv._conv_nd(
            var_396,
            self.var_401,
            bias=self.var_402,
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
        var_406 = paddle.nn.functional.conv._conv_nd(
            var_400,
            self.var_404,
            bias=self.var_405,
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
        var_407 = paddle.nn.functional.conv._conv_nd(
            var_365,
            self.var_369,
            bias=self.var_370,
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
        var_408 = paddle.nn.functional.activation.relu(var_407)
        var_409 = paddle.nn.functional.conv._conv_nd(
            var_365,
            self.var_373,
            bias=self.var_374,
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
        var_410 = paddle.nn.functional.activation.relu(var_409)
        var_411 = paddle.nn.functional.conv._conv_nd(
            var_408,
            self.var_377,
            bias=self.var_378,
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
        var_412 = paddle.nn.functional.activation.relu(var_411)
        var_413 = paddle.nn.functional.conv._conv_nd(
            var_410,
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
        var_414 = paddle.nn.functional.activation.relu(var_413)
        var_415 = paddle.nn.functional.conv._conv_nd(
            var_412,
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
        var_416 = paddle.nn.functional.activation.relu(var_415)
        var_417 = paddle.nn.functional.conv._conv_nd(
            var_414,
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
        var_418 = paddle.nn.functional.activation.relu(var_417)
        var_419 = paddle.nn.functional.conv._conv_nd(
            var_416,
            self.var_393,
            bias=self.var_394,
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
        var_420 = paddle.nn.functional.activation.relu(var_419)
        var_421 = paddle.nn.functional.conv._conv_nd(
            var_418,
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
        var_422 = paddle.nn.functional.activation.relu(var_421)
        var_423 = paddle.nn.functional.conv._conv_nd(
            var_420,
            self.var_401,
            bias=self.var_402,
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
        var_424 = paddle.nn.functional.conv._conv_nd(
            var_422,
            self.var_404,
            bias=self.var_405,
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
        var_425 = paddle.nn.functional.conv._conv_nd(
            var_366,
            self.var_369,
            bias=self.var_370,
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
        var_426 = paddle.nn.functional.activation.relu(var_425)
        var_427 = paddle.nn.functional.conv._conv_nd(
            var_366,
            self.var_373,
            bias=self.var_374,
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
        var_428 = paddle.nn.functional.activation.relu(var_427)
        var_429 = paddle.nn.functional.conv._conv_nd(
            var_426,
            self.var_377,
            bias=self.var_378,
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
        var_430 = paddle.nn.functional.activation.relu(var_429)
        var_431 = paddle.nn.functional.conv._conv_nd(
            var_428,
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
        var_432 = paddle.nn.functional.activation.relu(var_431)
        var_433 = paddle.nn.functional.conv._conv_nd(
            var_430,
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
        var_434 = paddle.nn.functional.activation.relu(var_433)
        var_435 = paddle.nn.functional.conv._conv_nd(
            var_432,
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
        var_436 = paddle.nn.functional.activation.relu(var_435)
        var_437 = paddle.nn.functional.conv._conv_nd(
            var_434,
            self.var_393,
            bias=self.var_394,
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
        var_438 = paddle.nn.functional.activation.relu(var_437)
        var_439 = paddle.nn.functional.conv._conv_nd(
            var_436,
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
        var_440 = paddle.nn.functional.activation.relu(var_439)
        var_441 = paddle.nn.functional.conv._conv_nd(
            var_438,
            self.var_401,
            bias=self.var_402,
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
        var_442 = paddle.nn.functional.conv._conv_nd(
            var_440,
            self.var_404,
            bias=self.var_405,
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
        var_443 = paddle.nn.functional.conv._conv_nd(
            var_367,
            self.var_369,
            bias=self.var_370,
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
        var_444 = paddle.nn.functional.activation.relu(var_443)
        var_445 = paddle.nn.functional.conv._conv_nd(
            var_367,
            self.var_373,
            bias=self.var_374,
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
        var_446 = paddle.nn.functional.activation.relu(var_445)
        var_447 = paddle.nn.functional.conv._conv_nd(
            var_444,
            self.var_377,
            bias=self.var_378,
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
        var_448 = paddle.nn.functional.activation.relu(var_447)
        var_449 = paddle.nn.functional.conv._conv_nd(
            var_446,
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
        var_450 = paddle.nn.functional.activation.relu(var_449)
        var_451 = paddle.nn.functional.conv._conv_nd(
            var_448,
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
        var_452 = paddle.nn.functional.activation.relu(var_451)
        var_453 = paddle.nn.functional.conv._conv_nd(
            var_450,
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
        var_454 = paddle.nn.functional.activation.relu(var_453)
        var_455 = paddle.nn.functional.conv._conv_nd(
            var_452,
            self.var_393,
            bias=self.var_394,
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
        var_456 = paddle.nn.functional.activation.relu(var_455)
        var_457 = paddle.nn.functional.conv._conv_nd(
            var_454,
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
        var_458 = paddle.nn.functional.activation.relu(var_457)
        var_459 = paddle.nn.functional.conv._conv_nd(
            var_456,
            self.var_401,
            bias=self.var_402,
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
        var_460 = paddle.nn.functional.conv._conv_nd(
            var_458,
            self.var_404,
            bias=self.var_405,
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
        var_461 = paddle.nn.functional.conv._conv_nd(
            var_368,
            self.var_369,
            bias=self.var_370,
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
        var_462 = paddle.nn.functional.activation.relu(var_461)
        var_463 = paddle.nn.functional.conv._conv_nd(
            var_368,
            self.var_373,
            bias=self.var_374,
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
        var_464 = paddle.nn.functional.activation.relu(var_463)
        var_465 = paddle.nn.functional.conv._conv_nd(
            var_462,
            self.var_377,
            bias=self.var_378,
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
        var_466 = paddle.nn.functional.activation.relu(var_465)
        var_467 = paddle.nn.functional.conv._conv_nd(
            var_464,
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
        var_468 = paddle.nn.functional.activation.relu(var_467)
        var_469 = paddle.nn.functional.conv._conv_nd(
            var_466,
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
        var_470 = paddle.nn.functional.activation.relu(var_469)
        var_471 = paddle.nn.functional.conv._conv_nd(
            var_468,
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
        var_472 = paddle.nn.functional.activation.relu(var_471)
        var_473 = paddle.nn.functional.conv._conv_nd(
            var_470,
            self.var_393,
            bias=self.var_394,
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
        var_474 = paddle.nn.functional.activation.relu(var_473)
        var_475 = paddle.nn.functional.conv._conv_nd(
            var_472,
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
        var_476 = paddle.nn.functional.activation.relu(var_475)
        var_477 = paddle.nn.functional.conv._conv_nd(
            var_474,
            self.var_401,
            bias=self.var_402,
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
        var_478 = paddle.nn.functional.conv._conv_nd(
            var_476,
            self.var_404,
            bias=self.var_405,
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
        return (
            var_403,
            var_423,
            var_441,
            var_459,
            var_477,
            var_406,
            var_424,
            var_442,
            var_460,
            var_478,
        )


class TestSIR34(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 256, 84, 128], dtype=paddle.float32),
            paddle.rand(shape=[1, 256, 42, 64], dtype=paddle.float32),
            paddle.rand(shape=[1, 256, 21, 32], dtype=paddle.float32),
            paddle.rand(shape=[1, 256, 11, 16], dtype=paddle.float32),
            paddle.rand(shape=[1, 256, 6, 8], dtype=paddle.float32),
        )
        self.net = SIR34()

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
