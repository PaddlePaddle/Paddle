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
from paddle.fluid import core
from paddle.vision.models import ResNet
from paddle.vision.models.resnet import BottleneckBlock, BasicBlock

paddle.enable_static()


class TestFuseResNetUnit(unittest.TestCase):
    def test_fuse_resenet_unit(self):
        program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.amp.fp16_guard():
            with paddle.static.program_guard(program, startup_program):
                x = paddle.static.data("x", [-1, 3, 224, 224])
                resnet50 = ResNet(BottleneckBlock, 50)
                out = resnet50(x)
        graph = core.Graph(program.desc)
        core.get_pass("fuse_resenet_unit").apply(graph)
        after_program = paddle.fluid.framework.IrGraph(graph).to_program()
        paddle.static.amp.cast_model_to_fp16(after_program)
        exe = paddle.static.Executor()
        exe.run(startup_program)
        feed = {"x": np.random.random([5, 3, 224, 224]).astype("float16")}
        exe.run(after_program, feed=feed, fetch_list=[out.name])
