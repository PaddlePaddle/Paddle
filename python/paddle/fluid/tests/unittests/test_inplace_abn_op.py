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

from __future__ import print_function

import unittest
import numpy as np
import os
import six
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid import compiler
import paddle.fluid.unique_name as unique_name
import paddle


class TestInplaceANBOpTraining(unittest.TestCase):
    def setUp(self):
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.N = 4
        self.C = 5
        self.H = 7
        self.W = 9
        self.dshape = [self.N, self.C, self.H, self.W]

    def build_program(self,
                      place,
                      layout,
                      seed,
                      only_forward=False,
                      activation="identity",
                      alpha=1.0,
                      use_cuda=False,
                      inplace=False):
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
                    stop_gradient=False)
                if inplace:
                    bn = fluid.layers.inplace_abn(
                        data,
                        act=activation,
                        param_attr=fluid.ParamAttr(name='bn_scale'),
                        bias_attr=fluid.ParamAttr(name='bn_bias'),
                        moving_mean_name='bn_moving_mean',
                        moving_variance_name='bn_moving_variance',
                        data_layout=layout,
                        is_test=only_forward,
                        act_alpha=alpha)
                else:
                    bn = fluid.layers.batch_norm(
                        data,
                        param_attr=fluid.ParamAttr(name='bn_scale'),
                        bias_attr=fluid.ParamAttr(name='bn_bias'),
                        moving_mean_name='bn_moving_mean',
                        moving_variance_name='bn_moving_variance',
                        data_layout=layout,
                        is_test=only_forward,
                        in_place=inplace)
                    if activation == 'leaky_relu':
                        bn = fluid.layers.leaky_relu(bn, alpha)
                    if activation == 'elu':
                        bn = fluid.layers.elu(bn, alpha)

                # NOTE: in inplace mode input and output of bn
                # may have same name, multiply 1. to generate 
                # a new Variable for fetch
                bn = bn * 1.

                sigmoid = fluid.layers.sigmoid(bn)
                out = fluid.layers.reduce_sum(sigmoid)
                if not only_forward:
                    sgd_opt = fluid.optimizer.SGD(learning_rate=0.0)
                    sgd_opt.backward(out)
        return main, startup, [out, bn]

    def compare(self, place, layout, only_forward, activation, alpha, use_cuda):
        seed = 10
        os.environ['FLAGS_cudnn_deterministic'] = "1"
        data = np.random.random(size=self.dshape).astype(self.dtype) * 4. - 2

        fetch_outs = []
        fetch_names = []
        for inplace in [False, True]:
            main, startup, outs = self.build_program(
                place,
                layout,
                seed,
                only_forward,
                activation,
                alpha,
                inplace=inplace)
            exe = fluid.Executor(place)
            exe.run(startup)

            fetch_name = [v.name for v in outs] + [
                'bn_moving_mean', 'bn_moving_variance', 'bn_scale', 'bn_bias'
            ]
            if not only_forward:
                others = [
                    'inplace_abn_0.tmp_0' if inplace else 'batch_norm_0.tmp_0',
                    'inplace_abn_0.tmp_1' if inplace else 'batch_norm_0.tmp_1',
                    'bn_scale@GRAD',
                    'bn_bias@GRAD',
                    'input@GRAD',
                ]
                fetch_name += others
            for nm in fetch_name:
                fv = fluid.framework._get_var(str(nm), program=main)
                fv.persistable = True

            build_strategy = fluid.BuildStrategy()
            build_strategy.sync_batch_norm = use_cuda and \
                        fluid.core.get_cuda_device_count() > 1
            build_strategy.enable_inplace = inplace
            exec_strategy = fluid.ExecutionStrategy()
            exec_strategy.num_threads = 1 if os.name == 'nt' else 0
            comp_prog1 = compiler.CompiledProgram(main).with_data_parallel(
                outs[0].name if not only_forward else None,
                build_strategy=build_strategy,
                exec_strategy=exec_strategy)
            bn_fetches = exe.run(program=main,
                                 feed={'input': data},
                                 fetch_list=fetch_name)
            fetch_outs.append(bn_fetches)
            fetch_names.append(fetch_name)

        for bn_val, inplace_abn_val, name1, name2 in zip(*(
                fetch_outs + fetch_names)):
            self.assertTrue(
                np.allclose(
                    bn_val, inplace_abn_val, atol=1e-2),
                "Output (" + name1 + ":" + name2 +
                ") has diff on {} with {} layout and {} activation. \n".format(
                    place, layout, activation) + "\nBN     " + str(bn_val) +
                "\n" + "Inplace ABN " + str(inplace_abn_val))

    def test_op(self):
        use_cudas = [False, True] if core.is_compiled_with_cuda() else [False]
        #use_cudas = [False]
        for use_cuda in use_cudas:
            place = core.CUDAPlace(0) if use_cuda else core.CPUPlace()
            layouts = ["NCHW", "NHWC"]
            for layout in layouts:
                for activation, alpha in zip([None, 'elu', 'leaky_relu'],
                                             [0., 1., 0.02]):
                    for infer_only in [True, False]:
                        self.compare(place, layout, infer_only, activation,
                                     alpha, use_cuda)

    def test_all_branches(self):
        seed = 10
        os.environ['FLAGS_cudnn_deterministic'] = "1"
        data = np.random.random(size=self.dshape).astype(self.dtype) * 4. - 2
        use_cudas = [False, True] if core.is_compiled_with_cuda() else [False]
        alpha = 0.1
        layouts = ["NCHW", "NHWC"]
        for use_cuda in use_cudas:
            place = core.CUDAPlace(0) if use_cuda else core.CPUPlace()
            for layout in layouts:
                for activation in ['identity', 'leaky_relu']:
                    main, startup, outs = self.build_program(
                        place, layout, seed, False, activation, alpha, use_cuda,
                        True)
                    exe = fluid.Executor(place)
                    exe.run(startup)
                    exe.run(program=main, feed={'input': data})


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
