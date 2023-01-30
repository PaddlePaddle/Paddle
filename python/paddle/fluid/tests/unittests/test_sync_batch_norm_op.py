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

<<<<<<< HEAD
import os
import unittest

import numpy as np
from decorator_helper import prog_scope
from op_test import OpTest, _set_use_system_allocator

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.nn as nn
from paddle.fluid import Program, compiler, program_guard
=======
from __future__ import print_function

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

from op_test import OpTest, _set_use_system_allocator
from decorator_helper import prog_scope
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

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
<<<<<<< HEAD
        # self.dtype = np.float32
=======
        #self.dtype = np.float32
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.N = 8
        self.C = 16
        self.H = 32
        self.W = 32
        self.dshape = [self.N, self.C, self.H, self.W]
        self.atol = 1e-3

<<<<<<< HEAD
    def _build_program(
        self, place, layout, seed, sync_bn=False, only_forward=False
    ):
=======
    def _build_program(self,
                       place,
                       layout,
                       seed,
                       sync_bn=False,
                       only_forward=False):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        """Build program."""
        main = fluid.Program()
        startup = fluid.Program()
        main.random_seed = seed
        startup.random_seed = seed
        use_cudnn = self.dtype == np.float16
        with fluid.unique_name.guard():
            with fluid.program_guard(main, startup):
<<<<<<< HEAD
                data = paddle.static.data(
                    name='input',
                    shape=self.dshape,
                    dtype=self.dtype,
                )
                data.desc.set_need_check_feed(False)
                conv = paddle.static.nn.conv2d(
=======
                data = fluid.layers.data(name='input',
                                         shape=self.dshape,
                                         dtype=self.dtype,
                                         append_batch_size=False)
                conv = fluid.layers.conv2d(
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    input=data,
                    num_filters=32,
                    filter_size=1,
                    param_attr=fluid.ParamAttr(name='conv2d_weight'),
                    bias_attr=False,
<<<<<<< HEAD
                    use_cudnn=use_cudnn,
                )
                bn = paddle.static.nn.batch_norm(
=======
                    use_cudnn=use_cudnn)
                bn = fluid.layers.batch_norm(
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    conv,
                    param_attr=fluid.ParamAttr(name='bn_scale'),
                    bias_attr=fluid.ParamAttr(name='bn_bias'),
                    moving_mean_name='bn_moving_mean',
                    moving_variance_name='bn_moving_variance',
                    data_layout=layout,
<<<<<<< HEAD
                    is_test=only_forward,
                )
=======
                    is_test=only_forward)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                if core.is_compiled_with_rocm():
                    bn = fluid.layers.cast(bn, 'float32')
                else:
                    bn = fluid.layers.cast(bn, 'float64')
<<<<<<< HEAD
                sigmoid = paddle.nn.functional.sigmoid(bn)
                out = paddle.sum(sigmoid)
=======
                sigmoid = fluid.layers.sigmoid(bn)
                out = fluid.layers.reduce_sum(sigmoid)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                if not sync_bn:
                    out = out / core.get_cuda_device_count()
                if not only_forward:
                    sgd_opt = fluid.optimizer.SGD(learning_rate=0.0)
                    sgd_opt.backward(out)
        return main, startup, [out, conv, bn]

    @prog_scope()
    def _compare(self, place, layout, only_forward):
        """Compare results."""
        seed = 10
        os.environ['FLAGS_cudnn_deterministic'] = "1"
        scope = core.Scope()
<<<<<<< HEAD
        data = np.random.random(size=self.dshape).astype(self.dtype) * 4.0 - 2
        data = create_or_get_tensor(
            scope, "input", OpTest.np_dtype_to_fluid_dtype(data), place
        )

        # Single-GPU, N = 32 per GPU
        main, startup, outs = self._build_program(
            place, layout, seed, False, only_forward
        )
        exe = fluid.Executor(place)
        exe.run(startup)
        fetch_names = [v.name for v in outs] + [
            'bn_moving_mean',
            'bn_moving_variance',
            'bn_scale',
            'bn_bias',
        ]
        if not only_forward:
            others = [
                'batch_norm_0.tmp_0',
                'batch_norm_0.tmp_1',
                'bn_scale@GRAD',
                'bn_bias@GRAD',
                'batch_norm_0.tmp_3@GRAD',
                'conv2d_0.tmp_0@GRAD',
            ]
            fetch_names += others
        bn_fetches = exe.run(
            program=main, feed={'input': data}, fetch_list=fetch_names
        )
=======
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
                'bn_bias@GRAD', 'batch_norm_0.tmp_3@GRAD', 'conv2d_0.tmp_0@GRAD'
            ]
            fetch_names += others
        bn_fetches = exe.run(program=main,
                             feed={'input': data},
                             fetch_list=fetch_names)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        #####################################################################
        # Multi-GPUs, self.N / core.get_cuda_device_count() per GPU
        assert core.get_cuda_device_count() > 1
<<<<<<< HEAD
        main, startup, outs = self._build_program(
            place, layout, seed, True, only_forward
        )
        exe = fluid.Executor(place)
        exe.run(startup)
        fetch_names = [v.name for v in outs] + [
            'bn_moving_mean',
            'bn_moving_variance',
            'bn_scale',
            'bn_bias',
        ]
        if not only_forward:
            others = [
                'batch_norm_0.tmp_0',
                'batch_norm_0.tmp_1',
                'bn_scale@GRAD',
                'bn_bias@GRAD',
                'batch_norm_0.tmp_3@GRAD',
                'conv2d_0.tmp_0@GRAD',
=======
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
            build_strategy=build_strategy,
        )
        sync_bn_fetches = exe.run(
            program=comp_prog, feed={'input': data}, fetch_list=fetch_names
        )

        for i in range(1, len(sync_bn_fetches)):
            bn_val = bn_fetches[i]
            sync_bn_val = sync_bn_fetches[i]
            if sync_bn_val.shape != bn_val.shape:
                sync_bn_val = sync_bn_val[: bn_val.shape[0]]
            np.testing.assert_allclose(
                bn_val,
                sync_bn_val,
                rtol=1e-05,
                atol=self.atol,
                err_msg='Output ('
                + fetch_names[i]
                + ') has diff. \n'
                + '\nBN     '
                + str(bn_val)
                + '\n'
                + 'Sync BN '
                + str(sync_bn_val),
            )
=======
            build_strategy=build_strategy)
        sync_bn_fetches = exe.run(program=comp_prog,
                                  feed={'input': data},
                                  fetch_list=fetch_names)

        for i in six.moves.xrange(1, len(sync_bn_fetches)):
            bn_val = bn_fetches[i]
            sync_bn_val = sync_bn_fetches[i]
            if sync_bn_val.shape != bn_val.shape:
                sync_bn_val = sync_bn_val[:bn_val.shape[0]]
            np.testing.assert_allclose(bn_val,
                                       sync_bn_val,
                                       rtol=1e-05,
                                       atol=self.atol,
                                       err_msg='Output (' + fetch_names[i] +
                                       ') has diff. \n' + '\nBN     ' +
                                       str(bn_val) + '\n' + 'Sync BN ' +
                                       str(sync_bn_val))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

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
        self.N = 8
        self.C = 16
        self.H = 32
        self.W = 32
        self.dshape = [self.N, self.C, self.H, self.W]
        self.atol = 1e-2


class TestDygraphSyncBatchNormAPIError(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_errors(self):
        if not core.is_compiled_with_cuda():
            return

        with program_guard(Program(), Program()):
            my_sync_batch_norm = paddle.nn.SyncBatchNorm(10)
<<<<<<< HEAD
            x1 = fluid.create_lod_tensor(
                np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], fluid.CUDAPlace(0)
            )
=======
            x1 = fluid.create_lod_tensor(np.array([-1, 3, 5, 5]),
                                         [[1, 1, 1, 1]], fluid.CUDAPlace(0))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.assertRaises(TypeError, my_sync_batch_norm, x1)

            # the input dtype of SyncBatchNorm must be float16 or float32 or float64
            # float16 only can be set on GPU place
<<<<<<< HEAD
            x2 = paddle.static.data(
                name='x2', shape=[-1, 3, 4, 5, 6], dtype="int32"
            )
            x2.desc.set_need_check_feed(False)
=======
            x2 = fluid.layers.data(name='x2', shape=[3, 4, 5, 6], dtype="int32")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.assertRaises(TypeError, my_sync_batch_norm, x2)


class TestConvertSyncBatchNorm(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_convert(self):
        if not core.is_compiled_with_cuda():
            return

        with program_guard(Program(), Program()):
<<<<<<< HEAD
            compare_model = paddle.nn.Sequential(
                paddle.nn.Conv2D(3, 5, 3),
                paddle.nn.BatchNorm2D(5),
                paddle.nn.BatchNorm2D(5),
            )
            model = paddle.nn.Sequential(
                paddle.nn.Conv2D(3, 5, 3),
                paddle.nn.BatchNorm2D(5),
                paddle.nn.BatchNorm2D(
                    5,
                    weight_attr=fluid.ParamAttr(name='bn.scale'),
                    bias_attr=fluid.ParamAttr(name='bn.bias'),
                ),
            )
=======
            compare_model = paddle.nn.Sequential(paddle.nn.Conv2D(3, 5, 3),
                                                 paddle.nn.BatchNorm2D(5),
                                                 paddle.nn.BatchNorm2D(5))
            model = paddle.nn.Sequential(
                paddle.nn.Conv2D(3, 5, 3), paddle.nn.BatchNorm2D(5),
                paddle.nn.BatchNorm2D(
                    5,
                    weight_attr=fluid.ParamAttr(name='bn.scale'),
                    bias_attr=fluid.ParamAttr(name='bn.bias')))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            for idx, sublayer in enumerate(compare_model.sublayers()):
                if isinstance(sublayer, paddle.nn.BatchNorm2D):
                    self.assertEqual(
<<<<<<< HEAD
                        isinstance(model[idx], paddle.nn.SyncBatchNorm), True
                    )


class TestConvertSyncBatchNormCast1(unittest.TestCase):
=======
                        isinstance(model[idx], paddle.nn.SyncBatchNorm), True)


class TestConvertSyncBatchNormCast1(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_convert(self):
        if not core.is_compiled_with_cuda():
            return

        class Net(nn.Layer):
<<<<<<< HEAD
            def __init__(self):
                super().__init__()
=======

            def __init__(self):
                super(Net, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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


class TestConvertSyncBatchNormCase2(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_convert(self):
        if not core.is_compiled_with_cuda():
            return

        with fluid.dygraph.guard(fluid.CUDAPlace(0)):

            class SyBNNet(paddle.nn.Layer):
<<<<<<< HEAD
                def __init__(self, in_ch=3, out_ch=3, dirate=1):
                    super().__init__()
=======

                def __init__(self, in_ch=3, out_ch=3, dirate=1):
                    super(SyBNNet, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    self.bn_s1 = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(
                        paddle.nn.BatchNorm3D(
                            out_ch,
                            weight_attr=paddle.ParamAttr(
<<<<<<< HEAD
                                regularizer=paddle.regularizer.L2Decay(0.0)
                            ),
                        )
                    )
                    self.bn_s2 = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(
                        paddle.nn.BatchNorm3D(out_ch, data_format='NDHWC')
                    )
=======
                                regularizer=paddle.regularizer.L2Decay(0.))))
                    self.bn_s2 = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(
                        paddle.nn.BatchNorm3D(out_ch, data_format='NDHWC'))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

                def forward(self, x):
                    x = self.bn_s1(x)
                    out = paddle.sum(paddle.abs(self.bn_s2(x)))
                    return out

            class BNNet(paddle.nn.Layer):
<<<<<<< HEAD
                def __init__(self, in_ch=3, out_ch=3, dirate=1):
                    super().__init__()
                    self.bn_s1 = paddle.nn.BatchNorm3D(
                        out_ch,
                        weight_attr=paddle.ParamAttr(
                            regularizer=paddle.regularizer.L2Decay(0.0)
                        ),
                    )
                    self.bn_s2 = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(
                        paddle.nn.BatchNorm3D(out_ch, data_format='NDHWC')
                    )
=======

                def __init__(self, in_ch=3, out_ch=3, dirate=1):
                    super(BNNet, self).__init__()
                    self.bn_s1 = paddle.nn.BatchNorm3D(
                        out_ch,
                        weight_attr=paddle.ParamAttr(
                            regularizer=paddle.regularizer.L2Decay(0.)))
                    self.bn_s2 = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(
                        paddle.nn.BatchNorm3D(out_ch, data_format='NDHWC'))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

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
            np.testing.assert_allclose(
                bn_out.numpy(),
                sybn_out.numpy(),
                rtol=1e-05,
<<<<<<< HEAD
                err_msg='Output has diff. \n'
                + '\nBN     '
                + str(bn_out.numpy())
                + '\n'
                + 'Sync BN '
                + str(sybn_out.numpy()),
            )


class TestDygraphSyncBatchNormDataFormatError(unittest.TestCase):
=======
                err_msg='Output has diff. \n' + '\nBN     ' +
                str(bn_out.numpy()) + '\n' + 'Sync BN ' + str(sybn_out.numpy()))


class TestDygraphSyncBatchNormDataFormatError(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_errors(self):
        if not core.is_compiled_with_cuda():
            return

        with fluid.dygraph.guard(fluid.CUDAPlace(0)):
            my_sync_batch_norm = paddle.nn.SyncBatchNorm(10, data_format='CN')
            data = np.random.random([3, 3, 3]).astype('float32')
            x = paddle.to_tensor(data)
            self.assertRaises(ValueError, my_sync_batch_norm, x)


if __name__ == '__main__':
    unittest.main()
