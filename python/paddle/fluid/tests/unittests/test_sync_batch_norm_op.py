#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler


class TestSyncBatchNormOpTraining(unittest.TestCase):
    def setUp(self):
        self.dtype = np.float32
        self.N = 32
        self.C = 16
        self.H = 56
        self.W = 64
        self.dshape = [self.N, self.C, self.H, self.W]

    def build_program(self, place, layout, seed, sync_bn=False):
        main = fluid.Program()
        startup = fluid.Program()
        main.random_seed = seed
        startup.random_seed = seed
        #with fluid.unique_name.guard():
        with fluid.program_guard(main, startup):
            data = fluid.layers.data(
                name='input',
                shape=self.dshape,
                dtype=self.dtype,
                append_batch_size=False)
            conv = fluid.layers.conv2d(
                input=data,
                num_filters=32,
                filter_size=1,
                param_attr=fluid.ParamAttr(name='conv2d_weight'),
                act='relu',
                bias_attr=False)
            if sync_bn:
                bn = fluid.layers.sync_batch_norm(
                    conv,
                    param_attr=fluid.ParamAttr(name='bn_scale'),
                    bias_attr=fluid.ParamAttr(name='bn_bias'),
                    moving_mean_name='bn_moving_mean_sync',
                    moving_variance_name='bn_moving_variance_sync',
                    data_layout=layout)
            else:
                bn = fluid.layers.batch_norm(
                    conv,
                    param_attr=fluid.ParamAttr(name='bn_scale'),
                    bias_attr=fluid.ParamAttr(name='bn_bias'),
                    moving_mean_name='bn_moving_mean_async',
                    moving_variance_name='bn_moving_variance_async',
                    data_layout=layout)
            out = fluid.layers.reduce_sum(bn)
            sgd_opt = fluid.optimizer.SGD(learning_rate=0.01)
            sgd_opt.backward(out)
            print(bn.name)
        return main, startup, [out, conv, bn]

    #def compare_output(self, a, b):

    def compare(self, place, layout):
        seed = 10
        np.random.seed(5)
        data = np.random.random(size=self.dshape).astype(self.dtype)

        # Single-GPU, N = 32 per GPU
        main, startup, outs = self.build_program(
            place, layout, seed, sync_bn=False)
        exe = fluid.Executor(place)
        exe.run(startup)
        fces = [v.name for v in outs] + [
            'bn_moving_mean_async', 'bn_moving_variance_async',
            'batch_norm_0.tmp_0', 'batch_norm_0.tmp_1', 'bn_scale', 'bn_bias',
            'batch_norm_0.tmp_2'
        ]
        fetches = exe.run(program=main, feed={'input': data}, fetch_list=fces)

        with open('one-gpu.txt', 'w+') as f:
            print(main, file=f)
        print(np.sum(np.abs(fetches[-1])))
        for nm, v in zip(fces, fetches):
            print(nm, v.shape, np.sum(np.abs(v)))

        # Multi-GPUs, N = 32 per GPU
        main, startup, outs = self.build_program(
            place, layout, seed, sync_bn=True)
        exe = fluid.Executor(place)
        exe.run(startup)

        with open('sync-gpu.txt', 'w+') as f:
            print(main, file=f)
        comp_prog = compiler.CompiledProgram(main).with_data_parallel(outs[0]
                                                                      .name)
        fces = [v.name for v in outs] + [
            'bn_moving_mean_sync', 'bn_moving_variance_sync',
            'sync_batch_norm_0.tmp_0', 'sync_batch_norm_0.tmp_1', 'bn_scale',
            'bn_bias', 'sync_batch_norm_0.tmp_2'
        ]
        fetches = exe.run(program=comp_prog,
                          feed={'input': data},
                          fetch_list=fces)
        for i in xrange(0, 3):
            print(fces[i], fetches[i].shape, np.sum(np.abs(fetches[i])))
        for i in xrange(3, 9):
            print(fces[i], fetches[i].shape, np.sum(np.abs(fetches[i])) / 2)
        print(np.sum(np.abs(fetches[-1])) / 2)

    def test_check_output(self):
        places = [core.CUDAPlace(0)]
        for place in places:
            #for layout in ["NCHW", "NHWC"]:
            for layout in ["NCHW"]:
                self.compare(place, layout)


if __name__ == '__main__':
    unittest.main()
