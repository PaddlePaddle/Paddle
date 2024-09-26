# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
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
import paddle.incubate
from paddle.base import core

paddle.enable_static()
np.random.seed(0)


@unittest.skipIf(
    not paddle.is_compiled_with_cuda()
    or paddle.get_cudnn_version() < 8000
    or paddle.device.cuda.get_device_capability()[0] < 7
    or paddle.device.cuda.get_device_capability()[0] >= 9,
    "only support with cuda and cudnn version is at least 8.0 "
    "and device's compute capability is at least 7.0 and less than 9.0",
)
class TestFuseResNetUnit(unittest.TestCase):
    def test_fuse_resnet_unit(self):
        place = paddle.CUDAPlace(0)
        with paddle.pir_utils.OldIrGuard():
            program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.amp.fp16_guard():
                with paddle.static.program_guard(program, startup_program):
                    x = paddle.static.data("x", [1, 64, 64, 8], dtype="float16")
                    conv2d = paddle.nn.Conv2D(
                        8, 32, 1, bias_attr=False, data_format='NHWC'
                    )
                    batch_norm = paddle.nn.BatchNorm(
                        32, act='relu', data_layout='NHWC'
                    )
                    out = batch_norm(conv2d(x))
            graph = core.Graph(program.desc)
            core.get_pass("fuse_resnet_unit").apply(graph)
            after_program = paddle.base.framework.IrGraph(graph).to_program()
            params = paddle.static.amp.cast_model_to_fp16(program)
            after_params = paddle.static.amp.cast_model_to_fp16(after_program)
            exe = paddle.static.Executor(place)
            exe.run(startup_program)
            paddle.static.amp.cast_parameters_to_fp16(
                place, program, to_fp16_var_names=params
            )
            paddle.static.amp.cast_parameters_to_fp16(
                place, after_program, to_fp16_var_names=after_params
            )
            feed = {"x": np.random.randn(1, 64, 64, 8).astype("float16")}
            before_out = exe.run(program, feed=feed, fetch_list=[out])
            after_out = exe.run(after_program, feed=feed, fetch_list=[out])
            np.testing.assert_allclose(
                before_out[0], after_out[0], rtol=1e-05, atol=0.005
            )


if __name__ == '__main__':
    unittest.main()
