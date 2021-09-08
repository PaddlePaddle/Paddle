#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import sys
sys.path.append("..")
import unittest
import numpy as np
import os
import six
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
import paddle.nn as nn
from paddle.fluid import compiler
from paddle.fluid import Program, program_guard

from paddle.fluid.tests.unittests.op_test import OpTest, _set_use_system_allocator
from paddle.fluid.tests.unittests.test_sync_batch_norm_op import create_or_get_tensor

_set_use_system_allocator(False)
paddle.enable_static()


@unittest.skipIf(False, "skip for tmp")
class TestSyncBatchNormOpTraining(unittest.TestCase):
    """sync_batch_norm op test."""

    def setUp(self):
        """Setup."""
        self.dtype = np.float32
        self.N = 8
        self.C = 16
        self.H = 32
        self.W = 32
        self.dshape = [self.N, self.C, self.H, self.W]
        self.atol = 1e-2

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
        use_cudnn = False
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
                if self.dtype == np.float16:
                    bn = fluid.layers.cast(bn, 'float32')
                sigmoid = fluid.layers.sigmoid(bn)
                out = fluid.layers.reduce_sum(sigmoid)
                if not sync_bn:
                    out = out / core.get_npu_device_count()
                if not only_forward:
                    sgd_opt = fluid.optimizer.SGD(learning_rate=0.0)
                    sgd_opt.backward(out)
        return main, startup, [out, conv, bn]

    def _compare(self, place, layout, only_forward):
        """Compare results."""
        seed = 10
        scope = core.Scope()
        data = np.random.random(size=self.dshape).astype(self.dtype) * 4. - 2
        data = create_or_get_tensor(scope, "input",
                                    OpTest.np_dtype_to_fluid_dtype(data), place)

        # Single-NPU, N = 32 per NPU
        main, startup, outs = self._build_program(place, layout, seed, False,
                                                  only_forward)
        print('main: ', main)
        exe = fluid.Executor(place)
        exe.run(startup)
        fetch_names = [v.name for v in outs] + [
            'bn_moving_mean', 'bn_moving_variance', 'bn_scale', 'bn_bias'
        ]
        if not only_forward:
            others = [
                'batch_norm_0.tmp_0', 'batch_norm_0.tmp_1', 'bn_scale@GRAD',
                'bn_bias@GRAD', 'batch_norm_0.tmp_3@GRAD', 'conv2d_0.tmp_0@GRAD'
            ]
            fetch_names += others
        bn_fetches = exe.run(program=main,
                             feed={'input': data},
                             fetch_list=fetch_names)

        #####################################################################
        # Multi-NPUs, self.N / core.get_npu_device_count() per NPU
        # return

        assert core.get_npu_device_count() > 1
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
                'bn_bias@GRAD', 'batch_norm_0.tmp_3@GRAD', 'conv2d_0.tmp_0@GRAD'
            ]
            fetch_names += others
        for nm in fetch_names:
            fv = fluid.framework._get_var(str(nm), program=main)
            fv.persistable = True
        # build_strategy = fluid.BuildStrategy()
        # build_strategy.sync_batch_norm = True
        # build_strategy.enable_inplace = False
        # build_strategy.memory_optimize = False
        # comp_prog = compiler.CompiledProgram(main).with_data_parallel(
        #     outs[0].name if not only_forward else None,
        #     build_strategy=build_strategy)
        comp_prog = main
        print('before comp_prog: ', comp_prog)
        comp_prog.blocks[0].ops[1].desc.set_type('sync_batch_norm')
        comp_prog.blocks[0].ops[7].desc.set_type('sync_batch_norm_grad')
        print('after comp_prog: ', comp_prog)
        sync_bn_fetches = exe.run(program=comp_prog,
                                  feed={'input': data},
                                  fetch_list=fetch_names)

        for i in six.moves.xrange(1, len(sync_bn_fetches)):
            print('i: ', i)
            print('fetch_names[i]: ', fetch_names[i])

            bn_val = bn_fetches[i]
            sync_bn_val = sync_bn_fetches[i]
            if sync_bn_val.shape != bn_val.shape:
                sync_bn_val = sync_bn_val[:bn_val.shape[0]]

            if fetch_names[i] == 'bn_moving_variance':
                print('skip bn_moving_variance (VarianceOut)')

                print('bn_val: ', str(bn_val))
                print('sync_bn_val: ', str(sync_bn_val))

                continue

            if fetch_names[i] == 'batch_norm_0.tmp_1':
                print('skip batch_norm_0.tmp_1 (SavedVariance)')

                print('bn_val: ', str(bn_val))
                print('sync_bn_val: ', str(sync_bn_val))

                continue

            self.assertTrue(
                np.allclose(
                    bn_val, sync_bn_val, atol=self.atol),
                "Output (" + fetch_names[i] + ") has diff. \n" + "\nBN     " +
                str(bn_val) + "\n" + "Sync BN " + str(sync_bn_val))

    def test_train(self):
        """Test training."""
        # places = [core.NPUPlace(0)]

        place_id = int(os.getenv("FLAGS_selected_npus", "0"))
        print("use selected_npus:", place_id)
        place = paddle.NPUPlace(place_id)
        places = [place]

        for place in places:
            for layout in ["NCHW", "NHWC"]:
                self._compare(place, layout, False)

    def test_infer(self):
        """Test inference."""
        if True:
            return

        places = [core.NPUPlace(0)]
        for place in places:
            for layout in ["NCHW", "NHWC"]:
                self._compare(place, layout, True)


@unittest.skipIf(True, "skip for tmp")
class TestFP16SyncBatchNormOpTraining(TestSyncBatchNormOpTraining):
    """sync_batch_norm op test for FP16 input."""

    def setUp(self):
        """Setup."""
        self.dtype = np.float16
        self.N = 8
        self.C = 16
        self.H = 32
        self.W = 32
        self.dshape = [self.N, self.C, self.H, self.W]
        self.atol = 1e-2


@unittest.skipIf(True, "skip for tmp")
class TestDygraphSyncBatchNormAPIError(unittest.TestCase):
    def test_errors(self):
        if not core.is_compiled_with_npu():
            return

        with program_guard(Program(), Program()):
            my_sync_batch_norm = paddle.nn.SyncBatchNorm(10)
            x1 = fluid.create_lod_tensor(
                np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], fluid.NPUPlace(0))
            self.assertRaises(TypeError, my_sync_batch_norm, x1)

            # the input dtype of SyncBatchNorm must be float16 or float32 or float64
            # float16 only can be set on GPU place
            x2 = fluid.layers.data(name='x2', shape=[3, 4, 5, 6], dtype="int32")
            self.assertRaises(TypeError, my_sync_batch_norm, x2)


@unittest.skipIf(True, "skip for tmp")
class TestConvertSyncBatchNorm(unittest.TestCase):
    def test_convert(self):
        if not core.is_compiled_with_npu():
            return

        with program_guard(Program(), Program()):
            compare_model = paddle.nn.Sequential(
                paddle.nn.Conv2D(3, 5, 3),
                paddle.nn.BatchNorm2D(5), paddle.nn.BatchNorm2D(5))
            model = paddle.nn.Sequential(
                paddle.nn.Conv2D(3, 5, 3),
                paddle.nn.BatchNorm2D(5),
                paddle.nn.BatchNorm2D(
                    5,
                    weight_attr=fluid.ParamAttr(name='bn.scale'),
                    bias_attr=fluid.ParamAttr(name='bn.bias')))
            model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            for idx, sublayer in enumerate(compare_model.sublayers()):
                if isinstance(sublayer, paddle.nn.BatchNorm2D):
                    self.assertEqual(
                        isinstance(model[idx], paddle.nn.SyncBatchNorm), True)


@unittest.skipIf(True, "skip for tmp")
class TestConvertSyncBatchNormCast1(unittest.TestCase):
    def test_convert(self):
        if not core.is_compiled_with_npu():
            return

        class Net(nn.Layer):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2D(3, 5, 3)
                self.bn = []
                bn = self.add_sublayer('bn', nn.BatchNorm2D(5))
                self.bn.append(bn)

            def forward(self, x):
                x = self.conv1(x)
                for bn in self.bn:
                    x = bn(x)
                return x

        model = nn.Sequential()
        model.add_sublayer('net1', Net())
        model.add_sublayer('net2', Net())
        compare_model = nn.Sequential()
        compare_model.add_sublayer('net1', Net())
        compare_model.add_sublayer('net2', Net())
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.assertEqual(len(compare_model.sublayers()), len(model.sublayers()))


@unittest.skipIf(True, "skip for tmp")
class TestConvertSyncBatchNormCase2(unittest.TestCase):
    def test_convert(self):
        if not core.is_compiled_with_npu():
            return

        with fluid.dygraph.guard(fluid.NPUPlace(0)):

            class SyBNNet(paddle.nn.Layer):
                def __init__(self, in_ch=3, out_ch=3, dirate=1):
                    super(SyBNNet, self).__init__()
                    self.bn_s1 = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(
                        paddle.nn.BatchNorm3D(
                            out_ch,
                            weight_attr=paddle.ParamAttr(
                                regularizer=paddle.regularizer.L2Decay(0.))))
                    self.bn_s2 = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(
                        paddle.nn.BatchNorm3D(
                            out_ch, data_format='NDHWC'))

                def forward(self, x):
                    x = self.bn_s1(x)
                    out = paddle.sum(paddle.abs(self.bn_s2(x)))
                    return out

            class BNNet(paddle.nn.Layer):
                def __init__(self, in_ch=3, out_ch=3, dirate=1):
                    super(BNNet, self).__init__()
                    self.bn_s1 = paddle.nn.BatchNorm3D(
                        out_ch,
                        weight_attr=paddle.ParamAttr(
                            regularizer=paddle.regularizer.L2Decay(0.)))
                    self.bn_s2 = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(
                        paddle.nn.BatchNorm3D(
                            out_ch, data_format='NDHWC'))

                def forward(self, x):
                    x = self.bn_s1(x)
                    out = paddle.sum(paddle.abs(self.bn_s2(x)))
                    return out

            bn_model = BNNet()
            sybn_model = SyBNNet()
            np.random.seed(10)
            data = np.random.random([3, 3, 3, 3, 3]).astype('float32')
            x = paddle.to_tensor(data)
            bn_out = bn_model(x)
            sybn_out = sybn_model(x)
            self.assertTrue(
                np.allclose(bn_out.numpy(), sybn_out.numpy()),
                "Output has diff. \n" + "\nBN     " + str(bn_out.numpy()) + "\n"
                + "Sync BN " + str(sybn_out.numpy()))


@unittest.skipIf(True, "skip for tmp")
class TestDygraphSyncBatchNormDataFormatError(unittest.TestCase):
    def test_errors(self):
        if not core.is_compiled_with_npu():
            return

        with fluid.dygraph.guard(fluid.NPUPlace(0)):
            my_sync_batch_norm = paddle.nn.SyncBatchNorm(10, data_format='CN')
            data = np.random.random([3, 3, 3]).astype('float32')
            x = paddle.to_tensor(data)
            self.assertRaises(ValueError, my_sync_batch_norm, x)


if __name__ == '__main__':
    unittest.main()
