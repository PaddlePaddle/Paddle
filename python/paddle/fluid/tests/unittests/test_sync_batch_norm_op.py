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

import os
import random
import subprocess
import tempfile
import unittest

import numpy as np
from decorator_helper import prog_scope
from eager_op_test import OpTest, _set_use_system_allocator

import paddle
from paddle import fluid, nn
from paddle.fluid import Program, core, program_guard

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
        # self.dtype = np.float32
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.N = 8
        self.C = 16
        self.H = 32
        self.W = 32
        self.dshape = [self.N, self.C, self.H, self.W]
        self.atol = 1e-3
        self.data_dir = tempfile.TemporaryDirectory()
        self.fleet_log_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.data_dir.cleanup()
        self.fleet_log_dir.cleanup()

    def multi_device_run(self, layout, fetch_list, only_forward=False):
        cmds = [
            "python",
            "-m",
            "paddle.distributed.launch",
        ]
        cmds += ["--log_dir", self.fleet_log_dir.name]
        cmds += ["dist_fleet_sync_batch_norm.py"]
        cmds += ["--data_dir", self.data_dir.name]

        dshape = [
            self.N // core.get_cuda_device_count(),
            self.C,
            self.H,
            self.W,
        ]
        cmds += ["--dshape", str(dshape)]
        cmds += ["--dtype", str(self.dtype.__name__)]
        cmds += ["--layout", layout]
        cmds += ["--fetch_list", str(fetch_list)]
        if only_forward:
            cmds += ["--only_forward"]
        if self.dtype == np.float16:
            cmds += ["--use_cudnn"]
        p = subprocess.run(cmds)
        assert p.returncode == 0, f"Fleet train: Failed: {p}"

    def _build_program(
        self, place, layout, seed, sync_bn=False, only_forward=False
    ):
        """Build program."""
        main = fluid.Program()
        startup = fluid.Program()
        main.random_seed = seed
        startup.random_seed = seed
        use_cudnn = self.dtype == np.float16
        with fluid.unique_name.guard():
            with fluid.program_guard(main, startup):
                data = paddle.static.data(
                    name='input',
                    shape=self.dshape,
                    dtype=self.dtype,
                )
                data.desc.set_need_check_feed(False)
                conv = paddle.static.nn.conv2d(
                    input=data,
                    num_filters=32,
                    filter_size=1,
                    param_attr=fluid.ParamAttr(name='conv2d_weight'),
                    bias_attr=False,
                    use_cudnn=use_cudnn,
                )
                bn = paddle.static.nn.batch_norm(
                    conv,
                    param_attr=fluid.ParamAttr(name='bn_scale'),
                    bias_attr=fluid.ParamAttr(name='bn_bias'),
                    moving_mean_name='bn_moving_mean',
                    moving_variance_name='bn_moving_variance',
                    data_layout=layout,
                    is_test=only_forward,
                )
                if core.is_compiled_with_rocm():
                    bn = paddle.cast(bn, 'float32')
                else:
                    bn = paddle.cast(bn, 'float64')
                sigmoid = paddle.nn.functional.sigmoid(bn)
                out = paddle.sum(sigmoid)
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
        paddle.enable_static()
        scope = core.Scope()
        data = np.random.random(size=self.dshape).astype(self.dtype) * 4.0 - 2
        stride = self.N // core.get_cuda_device_count()
        for id in range(core.get_cuda_device_count()):
            filepath = os.path.join(
                self.data_dir.name,
                'input_{}_{}_{}_{}.npy'.format(
                    id, only_forward, str(self.dtype.__name__), layout
                ),
            )
            np.save(filepath, data[id * stride : (id + 1) * stride])
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

        #####################################################################
        # Multi-GPUs, self.N / core.get_cuda_device_count() per GPU
        assert core.get_cuda_device_count() > 1

        fetch_names = [
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

        self.multi_device_run(
            layout, fetch_list=fetch_names, only_forward=only_forward
        )

        fetch_names = [v.name for v in outs] + fetch_names

        for i in range(1, len(bn_fetches)):
            bn_val = bn_fetches[i]
            file_path = os.path.join(
                self.data_dir.name,
                'output_{}_{}_{}_{}.npy'.format(
                    0, only_forward, self.dtype.__name__, i
                ),
            )
            sync_bn_val = np.load(file_path)
            if sync_bn_val.shape != bn_val.shape:
                bn_val = bn_val[:stride]
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

    def test_train(self):
        """Test training."""
        if not core.is_compiled_with_cuda():
            return

        places = [core.CUDAPlace(0)]
        for place in places:
            for layout in ["NHWC", "NCHW"]:
                self._compare(place, layout, False)

    def test_infer(self):
        """Test inference."""
        if not core.is_compiled_with_cuda():
            return

        places = [core.CUDAPlace(0)]
        for place in places:
            for layout in ["NHWC", "NCHW"]:
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
        self.data_dir = tempfile.TemporaryDirectory()
        self.fleet_log_dir = tempfile.TemporaryDirectory()


class TestDygraphSyncBatchNormAPIError(unittest.TestCase):
    def test_errors(self):
        if not core.is_compiled_with_cuda():
            return

        with program_guard(Program(), Program()):
            my_sync_batch_norm = paddle.nn.SyncBatchNorm(10)
            x1 = fluid.create_lod_tensor(
                np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], fluid.CUDAPlace(0)
            )
            self.assertRaises(TypeError, my_sync_batch_norm, x1)

            # the input dtype of SyncBatchNorm must be float16 or float32 or float64
            # float16 only can be set on GPU place
            x2 = paddle.static.data(
                name='x2', shape=[-1, 3, 4, 5, 6], dtype="int32"
            )
            x2.desc.set_need_check_feed(False)
            self.assertRaises(TypeError, my_sync_batch_norm, x2)


class TestConvertSyncBatchNorm(unittest.TestCase):
    def test_convert(self):
        if not core.is_compiled_with_cuda():
            return

        with program_guard(Program(), Program()):
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
            model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            for idx, sublayer in enumerate(compare_model.sublayers()):
                if isinstance(sublayer, paddle.nn.BatchNorm2D):
                    self.assertEqual(
                        isinstance(model[idx], paddle.nn.SyncBatchNorm), True
                    )


class TestConvertSyncBatchNormCast1(unittest.TestCase):
    def test_convert(self):
        if not core.is_compiled_with_cuda():
            return

        class Net(nn.Layer):
            def __init__(self):
                super().__init__()
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
    def test_convert(self):
        if not core.is_compiled_with_cuda():
            return

        with fluid.dygraph.guard(fluid.CUDAPlace(0)):

            class SyBNNet(paddle.nn.Layer):
                def __init__(self, in_ch=3, out_ch=3, dirate=1):
                    super().__init__()
                    self.bn_s1 = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(
                        paddle.nn.BatchNorm3D(
                            out_ch,
                            weight_attr=paddle.ParamAttr(
                                regularizer=paddle.regularizer.L2Decay(0.0)
                            ),
                        )
                    )
                    self.bn_s2 = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(
                        paddle.nn.BatchNorm3D(out_ch, data_format='NDHWC')
                    )

                def forward(self, x):
                    x = self.bn_s1(x)
                    out = paddle.sum(paddle.abs(self.bn_s2(x)))
                    return out

            class BNNet(paddle.nn.Layer):
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
                err_msg='Output has diff. \n'
                + '\nBN     '
                + str(bn_out.numpy())
                + '\n'
                + 'Sync BN '
                + str(sybn_out.numpy()),
            )


class TestDygraphSyncBatchNormDataFormatError(unittest.TestCase):
    def test_errors(self):
        if not core.is_compiled_with_cuda():
            return

        with fluid.dygraph.guard(fluid.CUDAPlace(0)):
            my_sync_batch_norm = paddle.nn.SyncBatchNorm(10, data_format='CN')
            data = np.random.random([3, 3, 3]).astype('float32')
            x = paddle.to_tensor(data)
            self.assertRaises(ValueError, my_sync_batch_norm, x)


if __name__ == '__main__':
    paddle.seed(0)
    np.random.seed(0)
    random.seed(0)
    unittest.main()
