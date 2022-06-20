#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import paddle
import paddle.static
import paddle.nn.functional as F
from paddle.fluid.tests.unittests.ipu.op_test_ipu import IPUOpTest, ExecutionModeFull


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestBase(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_data_feed()
        self.set_feed_attr()

    @property
    def fp16_enabled(self):
        return True

    def set_atol(self):
        self.atol = 1e-6
        self.rtol = 1e-6
        self.atol_fp16 = 1e-3
        self.rtol_fp16 = 1e-3

    def set_data_feed(self):
        data = np.random.uniform(size=[1, 10, 27, 27])
        self.feed_fp32 = {"in_0": data.astype(np.float32)}
        self.feed_fp16 = {"in_0": data.astype(np.float16)}

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def dtype_check(self, program, to_fp16_var_names):
        block = program.global_block()
        assert len(to_fp16_var_names) > 0
        for var_name in to_fp16_var_names:
            assert (block.var(var_name).dtype, paddle.float16)

    def _test_base(self, exec_mode):
        generator = paddle.fluid.unique_name.UniqueNameGenerator()
        scope = paddle.static.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = self.SEED
        startup_prog.random_seed = self.SEED

        with paddle.fluid.unique_name.guard(generator):
            with paddle.static.scope_guard(scope):
                with paddle.static.program_guard(main_prog, startup_prog):
                    x = paddle.static.data(
                        name=self.feed_list[0],
                        shape=self.feed_shape[0],
                        dtype='float32')

                    # using fp32
                    x = paddle.static.nn.conv2d(
                        input=x, num_filters=3, filter_size=3)
                    x = paddle.static.nn.batch_norm(x, act='relu')
                    x = F.max_pool2d(x, kernel_size=2, stride=2)

                    # using fp16
                    with paddle.static.amp.fp16_guard():
                        x = paddle.static.nn.conv2d(
                            input=x, num_filters=6, filter_size=3)
                        x = paddle.static.nn.batch_norm(x, act='relu')
                        x = F.max_pool2d(x, kernel_size=2, stride=2)

                    # using fp32
                    x = paddle.static.nn.fc(x, size=10)
                    loss = paddle.mean(x)
                    fetch_list = [loss.name]

                    if exec_mode == ExecutionModeFull.CPU_FP32:
                        place = paddle.CPUPlace()
                    else:
                        place = paddle.IPUPlace()

                    # cast model to fp16
                    if exec_mode == ExecutionModeFull.IPU_MIXED_PRECISION:
                        to_fp16_var_names = paddle.static.amp.cast_model_to_fp16(
                            main_prog, self.amp_list)
                        self.dtype_check(main_prog, to_fp16_var_names)

                    exe = paddle.static.Executor(place)
                    exe.run(startup_prog)

                    # cast parameters to fp16
                    if exec_mode == ExecutionModeFull.IPU_MIXED_PRECISION:
                        paddle.static.amp.cast_parameters_to_fp16(
                            paddle.CPUPlace(),
                            main_prog,
                            to_fp16_var_names=to_fp16_var_names)

                    if exec_mode != ExecutionModeFull.CPU_FP32:
                        ipu_strategy = paddle.static.IpuStrategy()
                        ipu_strategy.set_graph_config(is_training=False)
                        if exec_mode == ExecutionModeFull.IPU_POPART_FP16:
                            ipu_strategy.set_precision_config(enable_fp16=True)
                        program = paddle.static.IpuCompiledProgram(
                            main_prog, ipu_strategy=ipu_strategy).compile(
                                self.feed_list, fetch_list)
                    else:
                        program = main_prog

                    feed = self.feed_fp32
                    result = exe.run(program, feed=feed, fetch_list=fetch_list)
                    return result[0]

    def test(self):
        output_dict = {}
        for mode in ExecutionModeFull:
            if mode == ExecutionModeFull.IPU_POPART_FP16:
                continue
            if mode > ExecutionModeFull.IPU_FP32 and not self.fp16_enabled:
                break
            output_dict[mode] = self._test_base(mode).flatten()

        self.check(output_dict)


if __name__ == "__main__":
    unittest.main()
