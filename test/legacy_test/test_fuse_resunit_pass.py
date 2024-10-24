# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2023 NVIDIA Authors. All Rights Reserved.
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

import paddle
from paddle import nn


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


def verify_node_count(graph, node_name, target_count):
    count = 0
    for node in graph.nodes():
        if node.name() == node_name:
            count += 1
    return count == target_count


class ConvBNActLayer(paddle.nn.Layer):
    def __init__(self, num_channels, num_filters, filter_size):
        super().__init__()
        self.act = nn.ReLU()
        self.conv = nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=1,
            padding=(filter_size - 1) // 2,
            groups=1,
            bias_attr=False,
            data_format="NHWC",
        )
        self.bn = nn.BatchNorm(num_filters, data_layout="NHWC")

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class ResUnit(paddle.nn.Layer):
    def __init__(self, hidden, is_shortcut):
        super().__init__()
        self.is_shortcut = is_shortcut
        filter_size = 3
        num_channels = hidden
        num_filters = hidden
        self.conv_bn1 = ConvBNActLayer(hidden, hidden, filter_size)
        self.conv_bn2 = ConvBNActLayer(hidden, hidden, filter_size)
        self.act = nn.ReLU()
        self.conv1 = nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=1,
            padding=(filter_size - 1) // 2,
            groups=1,
            bias_attr=False,
            data_format="NHWC",
        )
        self.bn1 = nn.BatchNorm(num_filters, data_layout="NHWC")
        if not self.is_shortcut:
            self.conv2 = nn.Conv2D(
                in_channels=num_channels,
                out_channels=num_filters,
                kernel_size=filter_size,
                stride=1,
                padding=(filter_size - 1) // 2,
                groups=1,
                bias_attr=False,
                data_format="NHWC",
            )
            self.bn2 = nn.BatchNorm(num_filters, data_layout="NHWC")

    def forward(self, input):
        x1 = self.conv_bn1(input)
        x1 = self.conv_bn2(x1)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        if not self.is_shortcut:
            x2 = self.conv2(input)
            x2 = self.bn2(x2)
        else:
            x2 = input
        output = self.act(paddle.add(x1, x2))
        return output


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFuseResUnitBase(unittest.TestCase):
    def setUp(self):
        self.batch = 8
        self.hidden = 16
        self.width = 64
        self.height = 64
        self.iter = 5
        self.set_attr()

        paddle.enable_static()
        np.random.seed(10)
        paddle.seed(10)
        paddle.framework.random._manual_program_seed(10)

        self.place = paddle.CUDAPlace(0)
        self.exe = paddle.static.Executor(self.place)

        self.feeds = [
            {
                "input": np.random.random(
                    (self.batch, self.height, self.width, self.hidden)
                ).astype("float16")
                + i,
            }
            for i in range(self.iter)
        ]

    def build_program(
        self, main_program, startup_program, is_shortcut=True, is_training=False
    ):
        with paddle.static.program_guard(main_program, startup_program):
            with paddle.utils.unique_name.guard():
                x1 = paddle.static.data(
                    name="input",
                    shape=[-1, self.height, self.width, self.hidden],
                    dtype='float16',
                )
                layer1 = ConvBNActLayer(self.hidden, self.hidden, 3)
                resunit_layer1 = ResUnit(self.hidden, is_shortcut)
                resunit_layer2 = ResUnit(self.hidden, is_shortcut)
                layer2 = ConvBNActLayer(self.hidden, self.hidden, 3)
                optimizer = None
                with paddle.static.amp.fp16_guard():
                    out = layer1(x1)
                    out = resunit_layer1(out)
                    out = resunit_layer2(out)
                    out = layer2(out)
                    loss = paddle.mean(out)
                    if is_training:
                        optimizer = paddle.optimizer.SGD(learning_rate=0.001)
                        optimizer = paddle.static.amp.decorate(
                            optimizer=optimizer,
                            init_loss_scaling=128.0,
                            use_dynamic_loss_scaling=True,
                            use_pure_fp16=True,
                            use_fp16_guard=True,
                        )
                        optimizer.minimize(loss)
        return loss.name, optimizer

    def cal_output(self, enable_fusion):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        output_name, optimizer = self.build_program(
            main_prog, startup_prog, self.is_shortcut, self.is_training
        )
        loss_list = []
        scope = paddle.static.Scope()
        with paddle.static.scope_guard(scope):
            self.exe.run(startup_prog)
            if self.is_training:
                optimizer.amp_init(self.place, scope=scope)
            else:
                self._cast_model_to_fp16(main_prog)
            build_strategy = paddle.static.BuildStrategy()
            build_strategy.fuse_resunit = enable_fusion
            program = paddle.static.CompiledProgram(
                main_prog, build_strategy=build_strategy
            )

            for i in range(self.iter):
                result = self.exe.run(
                    program, feed=self.feeds[i], fetch_list=[output_name]
                )
                loss_list.append(result)

        if enable_fusion:
            self.assertTrue(
                verify_node_count(
                    program._graph, "fused_scale_bias_add_relu", 2
                ),
                f"[{type(self).__name__}] The number of fused_scale_bias_add_relu is miss-matched in the computing graph.",
            )
            conv_bnstats_count = 6 if self.is_shortcut else 8
            self.assertTrue(
                verify_node_count(
                    program._graph,
                    "fused_scale_bias_relu_conv_bn",
                    conv_bnstats_count,
                ),
                f"[{type(self).__name__}] The number of fused_scale_bias_relu_conv_bn is miss-matched in the computing graph.",
            )

        return np.array(loss_list)

    def _test_output(self):
        results_ref = self.cal_output(enable_fusion=False)
        results_actual = self.cal_output(enable_fusion=True)

        np.testing.assert_allclose(
            results_ref,
            results_actual,
            rtol=self.rtol,
            atol=self.atol,
            err_msg=f"[{type(self).__name__}] outputs are miss-matched.",
        )

    def set_attr(self):
        self.atol = 1e-4
        self.rtol = 1e-4

        self.is_shortcut = True
        self.is_training = False

    def _cast_model_to_fp16(self, prog):
        fp16_var_list = paddle.static.amp.cast_model_to_fp16(prog)
        paddle.static.amp.cast_parameters_to_fp16(
            self.place, prog, to_fp16_var_names=fp16_var_list
        )


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFuseResUnitShortcutFwd(TestFuseResUnitBase):
    def set_attr(self):
        self.atol = 1e-3
        self.rtol = 1e-2

        self.is_shortcut = True
        self.is_training = False

    def test_output(self):
        self._test_output()


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFuseResUnitDualFwd(TestFuseResUnitBase):
    def set_attr(self):
        self.atol = 1e-3
        self.rtol = 1e-2

        self.is_shortcut = False
        self.is_training = False

    def test_output(self):
        self._test_output()


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFuseResUnitBwd(TestFuseResUnitBase):
    def set_attr(self):
        self.atol = 1e-3
        self.rtol = 1e-2

        self.is_shortcut = True
        self.is_training = True

    def test_output(self):
        self._test_output()


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFuseResUnitDualBwd(TestFuseResUnitBase):
    def set_attr(self):
        self.atol = 1e-3
        self.rtol = 1e-2

        self.is_shortcut = False
        self.is_training = True

    def test_output(self):
        self._test_output()


if __name__ == "__main__":
    unittest.main()
