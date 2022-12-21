#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core


class TestInplaceANBOpTraining(unittest.TestCase):
    def setUp(self):
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.N = 4
        self.C = 5
        self.H = 7
        self.W = 9
        self.dshape = [self.N, self.C, self.H, self.W]

    def build_program(
        self,
        place,
        layout,
        seed,
        only_forward=False,
        activation="identity",
        alpha=1.0,
        use_cuda=False,
        inplace=False,
    ):
        main = fluid.Program()
        startup = fluid.Program()
        main.random_seed = seed
        startup.random_seed = seed
        with fluid.unique_name.guard():
            with fluid.program_guard(main, startup):
                data = fluid.layers.data(
                    name='input',
                    shape=self.dshape,
                    dtype=self.dtype,
                    append_batch_size=False,
                    stop_gradient=False,
                )

                bn = paddle.static.nn.batch_norm(
                    data,
                    param_attr=fluid.ParamAttr(name='bn_scale'),
                    bias_attr=fluid.ParamAttr(name='bn_bias'),
                    moving_mean_name='bn_moving_mean',
                    moving_variance_name='bn_moving_variance',
                    data_layout=layout,
                    is_test=only_forward,
                    in_place=inplace,
                )
                if activation == 'leaky_relu':
                    bn = paddle.nn.functional.leaky_relu(bn, alpha)
                if activation == 'elu':
                    bn = paddle.nn.functional.elu(bn, alpha)

                # NOTE: in inplace mode input and output of bn
                # may have same name, multiply 1. to generate
                # a new Variable for fetch
                bn = bn * 1.0
                sigmoid = paddle.nn.functional.sigmoid(bn)
                out = paddle.sum(sigmoid)
                if not only_forward:
                    sgd_opt = fluid.optimizer.SGD(learning_rate=0.0)
                    sgd_opt.backward(out)
        return main, startup, [out, bn]

    def test_all_branches(self):
        seed = 10
        os.environ['FLAGS_cudnn_deterministic'] = "1"
        data = np.random.random(size=self.dshape).astype(self.dtype) * 4.0 - 2
        use_cudas = [False, True] if core.is_compiled_with_cuda() else [False]
        alpha = 0.1
        layouts = ["NCHW", "NHWC"]
        for use_cuda in use_cudas:
            place = core.CUDAPlace(0) if use_cuda else core.CPUPlace()
            for layout in layouts:
                for activation in ['identity', 'leaky_relu']:
                    main, startup, outs = self.build_program(
                        place,
                        layout,
                        seed,
                        False,
                        activation,
                        alpha,
                        use_cuda,
                        False,
                    )
                    exe = fluid.Executor(place)
                    exe.run(startup)
                    exe.run(program=main, feed={'input': data})


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
