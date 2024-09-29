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

import os
import unittest

import numpy as np
from dist_pass_test_base import DistPassTestBase

import paddle
from paddle import ParamAttr, nn
from paddle.distributed import fleet
from paddle.distributed.passes import PassManager, new_pass
from paddle.nn import BatchNorm, Conv2D
from paddle.nn.initializer import Constant, KaimingNormal

paddle.enable_static()
np.random.seed(12345)
paddle.seed(12345)


def skip_unit_test():
    return (
        not paddle.is_compiled_with_cuda()
        or paddle.device.cuda.get_device_capability()[0] < 8
        or paddle.get_cudnn_version() < 8900
    )


skip_msg = (
    "only support with cuda and CUDNN 8.9 or later,"
    " and only Ampere or later devices are supported"
)


class ConvBNLayer(nn.Layer):
    def __init__(
        self,
        num_channels,
        num_filters,
        filter_size,
        stride=1,
        groups=1,
        act=None,
        lr_mult=1.0,
        data_format="NCHW",
        bn_weight_decay=True,
    ):
        super().__init__()
        self.act = act
        self.conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(
                learning_rate=lr_mult, initializer=KaimingNormal()
            ),
            bias_attr=False,
            data_format=data_format,
        )
        self.bn = BatchNorm(
            num_filters,
            param_attr=ParamAttr(
                learning_rate=lr_mult,
                regularizer=(
                    None if bn_weight_decay else paddle.regularizer.L2Decay(0.0)
                ),
                initializer=Constant(1.0),
            ),
            bias_attr=ParamAttr(
                learning_rate=lr_mult,
                regularizer=(
                    None if bn_weight_decay else paddle.regularizer.L2Decay(0.0)
                ),
                initializer=Constant(0.0),
            ),
            data_layout=data_format,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act:
            x = self.relu(x)
        return x


class BottleneckBlock(nn.Layer):
    def __init__(
        self,
        num_channels,
        num_filters,
        stride,
        shortcut=True,
        lr_mult=1.0,
        data_format="NCHW",
        bn_weight_decay=True,
    ):
        super().__init__()

        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act="relu",
            lr_mult=lr_mult,
            data_format=data_format,
            bn_weight_decay=bn_weight_decay,
        )
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act="relu",
            lr_mult=lr_mult,
            data_format=data_format,
            bn_weight_decay=bn_weight_decay,
        )
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None,
            lr_mult=lr_mult,
            data_format=data_format,
            bn_weight_decay=bn_weight_decay,
        )

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride,
                lr_mult=lr_mult,
                data_format=data_format,
                bn_weight_decay=bn_weight_decay,
            )
        self.relu = nn.ReLU()
        self.shortcut = shortcut

    def forward(self, x):
        identity = x
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        if self.shortcut:
            short = identity
        else:
            short = self.short(identity)
        x = paddle.add(x=x, y=short)
        x = self.relu(x)
        return x


class ResUnitNet(nn.Layer):
    def __init__(self, shortcut):
        super().__init__()
        self.shortcut = shortcut
        self.conv1 = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=3,
            act='relu',
            data_format='NHWC',
        )

        self.block = BottleneckBlock(
            num_channels=64,
            num_filters=16,
            stride=1,
            shortcut=self.shortcut,
            lr_mult=1.0,
            data_format="NHWC",
            bn_weight_decay=True,
        )

        self.conv2 = ConvBNLayer(
            num_channels=64,
            num_filters=64,
            filter_size=3,
            act='relu',
            data_format='NHWC',
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.block(out)
        out = self.conv2(out)
        out = paddle.flatten(out, 1)
        return out


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFuseResUnitPass(DistPassTestBase):
    def init(self):
        self.atol = 1e-2
        self.rtol = 1e-2
        paddle.set_flags({'FLAGS_conv_workspace_size_limit': 1000})
        self.init_attr()

    def init_attr(self):
        self.shortcut = True

    def get_model(self, place, batch_size=32, image_shape=[224, 224, 3]):
        image = paddle.static.data(
            shape=[batch_size, *image_shape], dtype='float32', name='image'
        )

        model = ResUnitNet(self.shortcut)
        pred_out = model(image)
        loss = paddle.mean(pred_out)
        optimizer = paddle.optimizer.Adam(learning_rate=1e-3)

        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.fuse_all_reduce_ops = False
        dist_strategy.without_graph_optimization = True
        dist_strategy.amp = True
        dist_strategy.amp_configs = {
            "init_loss_scaling": 128.0,
            "use_dynamic_loss_scaling": True,
        }
        build_strategy = paddle.static.BuildStrategy()
        settings = {
            "fuse_bn_act_ops": False,
            "fuse_bn_add_act_ops": False,
            "enable_inplace": False,
        }
        for k, v in settings.items():
            setattr(build_strategy, k, v)
        dist_strategy.build_strategy = build_strategy
        fleet.init(is_collective=True, strategy=dist_strategy)
        optimizer = fleet.distributed_optimizer(optimizer)
        optimizer.minimize(loss)

        rank = paddle.distributed.get_rank()

        def reader():
            seed = int(os.environ.get("SEED", 0))
            np.random.seed(seed + rank)
            for _ in range(10):
                image_np = np.random.random(size=image.shape).astype('float32')
                yield image_np,

        main_program = paddle.static.default_main_program()
        startup_program = paddle.static.default_startup_program()
        return main_program, startup_program, [image], [loss], reader

    def apply_passes(self, main_prog, startup_prog):
        pass_manager = PassManager([new_pass("fuse_resunit")])
        pass_manager.apply([main_prog], [startup_prog])
        print(pass_manager.names)

        op_type = []
        for op in main_prog.global_block().ops:
            op_type.append(op.type)
        self.assertTrue("fused_scale_bias_add_relu" in op_type)
        self.assertTrue("fused_scale_bias_relu_conv_bn" in op_type)
        self.assertTrue("fused_dconv_drelu_dbn" in op_type)

    def test_fuse_resunit(self):
        self.check_main()


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFuseResUnitPassDual(TestFuseResUnitPass):
    def init_attr(self):
        self.shortcut = False


if __name__ == "__main__":
    unittest.main()
