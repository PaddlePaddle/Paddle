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
"""
test for sync bachnorm op.
for both FP64 and FP16 input.
"""

from __future__ import print_function

import unittest
import numpy as np
import os
import six
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler

from op_test import OpTest, _set_use_system_allocator

_set_use_system_allocator(True)


def create_or_get_tensor(scope, var_name, var, place):
    """Get tensor, if not found, create a new one."""
    tensor = scope.var(var_name).get_tensor()
    if var is not None:
        assert isinstance(var, np.ndarray)
        tensor.set_recursive_sequence_lengths([])
        tensor.set(var, place)
    return tensor


class TestSyncBatchNormOpTraining(unittest.TestCase):
    """sync_batch_norm op test."""

    def setUp(self):
        """Setup."""
        #self.dtype = np.float32
        self.dtype = np.float64
        self.N = 32
        self.C = 16
        self.H = 64
        self.W = 32
        self.dshape = [self.N, self.C, self.H, self.W]
        self.atol = 1e-3

    def _build_program(self,
                       place,
                       layout,
                       seed,
                       sync_bn=False,
                       only_forward=False):
        """Build program."""
        main = fluid.Program()
        startup = fluid.Program()
        main.random_seed = seed
        startup.random_seed = seed
        use_cudnn = self.dtype == np.float16
        with fluid.unique_name.guard():
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
                    bias_attr=False,
                    use_cudnn=use_cudnn)
                bn = fluid.layers.batch_norm(
                    conv,
                    param_attr=fluid.ParamAttr(name='bn_scale'),
                    bias_attr=fluid.ParamAttr(name='bn_bias'),
                    moving_mean_name='bn_moving_mean',
                    moving_variance_name='bn_moving_variance',
                    data_layout=layout,
                    is_test=only_forward)
                bn = fluid.layers.cast(bn, 'float64')
                sigmoid = fluid.layers.sigmoid(bn)
                out = fluid.layers.reduce_sum(sigmoid)
                if not sync_bn:
                    out = out / core.get_cuda_device_count()
                if not only_forward:
                    sgd_opt = fluid.optimizer.SGD(learning_rate=0.0)
                    sgd_opt.backward(out)
        return main, startup, [out, conv, bn]

    def _compare(self, place, layout, only_forward):
        """Compare results."""
        seed = 10
        os.environ['FLAGS_cudnn_deterministic'] = "1"
        scope = core.Scope()
        data = np.random.random(size=self.dshape).astype(self.dtype) * 4. - 2
        data = create_or_get_tensor(scope, "input",
                                    OpTest.np_dtype_to_fluid_dtype(data), place)

        # Single-GPU, N = 32 per GPU
        main, startup, outs = self._build_program(place, layout, seed, False,
                                                  only_forward)
        exe = fluid.Executor(place)
        exe.run(startup)
        fetch_names = [v.name for v in outs] + [
            'bn_moving_mean', 'bn_moving_variance', 'bn_scale', 'bn_bias'
        ]
        if not only_forward:
            others = [
                'batch_norm_0.tmp_0', 'batch_norm_0.tmp_1', 'bn_scale@GRAD',
                'bn_bias@GRAD', 'batch_norm_0.tmp_2@GRAD', 'conv2d_0.tmp_0@GRAD'
            ]
            fetch_names += others
        bn_fetches = exe.run(program=main,
                             feed={'input': data},
                             fetch_list=fetch_names)

        #####################################################################
        # Multi-GPUs, self.N / core.get_cuda_device_count() per GPU
        assert core.get_cuda_device_count() > 1
        main, startup, outs = self._build_program(place, layout, seed, True,
                                                  only_forward)
        exe = fluid.Executor(place)
        exe.run(startup)
        fetch_names = [v.name for v in outs] + [
            'bn_moving_mean', 'bn_moving_variance', 'bn_scale', 'bn_bias'
        ]
        if not only_forward:
            others = [
                'batch_norm_0.tmp_0', 'batch_norm_0.tmp_1', 'bn_scale@GRAD',
                'bn_bias@GRAD', 'batch_norm_0.tmp_2@GRAD', 'conv2d_0.tmp_0@GRAD'
            ]
            fetch_names += others
        for nm in fetch_names:
            fv = fluid.framework._get_var(str(nm), program=main)
            fv.persistable = True
        build_strategy = fluid.BuildStrategy()
        build_strategy.sync_batch_norm = True
        build_strategy.enable_inplace = False
        build_strategy.memory_optimize = False
        comp_prog = compiler.CompiledProgram(main).with_data_parallel(
            outs[0].name if not only_forward else None,
            build_strategy=build_strategy)
        sync_bn_fetches = exe.run(program=comp_prog,
                                  feed={'input': data},
                                  fetch_list=fetch_names)

        for i in six.moves.xrange(1, len(sync_bn_fetches)):
            bn_val = bn_fetches[i]
            sync_bn_val = sync_bn_fetches[i]
            if sync_bn_val.shape != bn_val.shape:
                sync_bn_val = sync_bn_val[:bn_val.shape[0]]
            self.assertTrue(
                np.allclose(
                    bn_val, sync_bn_val, atol=self.atol),
                "Output (" + fetch_names[i] + ") has diff. \n" + "\nBN     " +
                str(bn_val) + "\n" + "Sync BN " + str(sync_bn_val))

    def test_train(self):
        """Test training."""
        if not core.is_compiled_with_cuda():
            return

        places = [core.CUDAPlace(0)]
        for place in places:
            for layout in ["NCHW", "NHWC"]:
                self._compare(place, layout, False)

    def test_infer(self):
        """Test inference."""
        if not core.is_compiled_with_cuda():
            return

        places = [core.CUDAPlace(0)]
        for place in places:
            for layout in ["NCHW", "NHWC"]:
                self._compare(place, layout, True)


class TestFP16SyncBatchNormOpTraining(TestSyncBatchNormOpTraining):
    """sync_batch_norm op test for FP16 input."""

    def setUp(self):
        """Setup."""
        self.dtype = np.float16
        self.N = 32
        self.C = 16
        self.H = 64
        self.W = 32
        self.dshape = [self.N, self.C, self.H, self.W]
        self.atol = 1e-2


if __name__ == '__main__':
    unittest.main()
