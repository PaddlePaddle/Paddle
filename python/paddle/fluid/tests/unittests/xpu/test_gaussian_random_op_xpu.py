#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import sys

sys.path.append("..")
import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper
import paddle

paddle.enable_static()


class XPUTestGaussianRandomOp(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'gaussian_random'
        self.use_dynamic_create_class = False

    class TestGaussianRandomOp(XPUOpTest):

        def init(self):
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.op_type = 'gaussian_random'

        def setUp(self):
            self.init()
            self.python_api = paddle.normal
            self.set_attrs()
            self.inputs = {}
            self.use_mkldnn = False
            self.attrs = {
                "shape": [123, 92],
                "mean": self.mean,
                "std": self.std,
                "seed": 10,
                "use_mkldnn": self.use_mkldnn
            }
            paddle.seed(10)

            self.outputs = {'Out': np.zeros((123, 92), dtype=self.dtype)}

        def set_attrs(self):
            self.mean = 1.0
            self.std = 2.

        def test_check_output(self):
            self.check_output_with_place_customized(self.verify_output,
                                                    self.place)

        def verify_output(self, outs):
            self.assertEqual(outs[0].shape, (123, 92))
            hist, _ = np.histogram(outs[0], range=(-3, 5))
            hist = hist.astype("float32")
            hist /= float(outs[0].size)
            data = np.random.normal(size=(123, 92), loc=1, scale=2)
            hist2, _ = np.histogram(data, range=(-3, 5))
            hist2 = hist2.astype("float32")
            hist2 /= float(outs[0].size)
            np.testing.assert_allclose(hist, hist2, rtol=0, atol=0.01)

    class TestMeanStdAreInt(TestGaussianRandomOp):

        def set_attrs(self):
            self.mean = 1
            self.std = 2

    # Situation 2: Attr(shape) is a list(with tensor)
    class TestGaussianRandomOp_ShapeTensorList(TestGaussianRandomOp):

        def setUp(self):
            '''Test gaussian_random op with specified value
            '''
            self.init()
            self.init_data()
            shape_tensor_list = []
            for index, ele in enumerate(self.shape):
                shape_tensor_list.append(("x" + str(index), np.ones(
                    (1)).astype('int32') * ele))

            self.attrs = {
                'shape': self.infer_shape,
                'mean': self.mean,
                'std': self.std,
                'seed': self.seed,
                'use_mkldnn': self.use_mkldnn
            }

            self.inputs = {"ShapeTensorList": shape_tensor_list}
            self.outputs = {'Out': np.zeros(self.shape, dtype=self.dtype)}

        def init_data(self):
            self.shape = [123, 92]
            self.infer_shape = [-1, 92]
            self.use_mkldnn = False
            self.mean = 1.0
            self.std = 2.0
            self.seed = 10

        def test_check_output(self):
            self.check_output_with_place_customized(self.verify_output,
                                                    self.place)

    class TestGaussianRandomOp2_ShapeTensorList(
            TestGaussianRandomOp_ShapeTensorList):

        def init_data(self):
            self.shape = [123, 92]
            self.infer_shape = [-1, -1]
            self.use_mkldnn = False
            self.mean = 1.0
            self.std = 2.0
            self.seed = 10

    class TestGaussianRandomOp3_ShapeTensorList(
            TestGaussianRandomOp_ShapeTensorList):

        def init_data(self):
            self.shape = [123, 92]
            self.infer_shape = [123, -1]
            self.use_mkldnn = True
            self.mean = 1.0
            self.std = 2.0
            self.seed = 10

    class TestGaussianRandomOp4_ShapeTensorList(
            TestGaussianRandomOp_ShapeTensorList):

        def init_data(self):
            self.shape = [123, 92]
            self.infer_shape = [123, -1]
            self.use_mkldnn = False
            self.mean = 1.0
            self.std = 2.0
            self.seed = 10

    # Situation 3: shape is a tensor
    class TestGaussianRandomOp1_ShapeTensor(TestGaussianRandomOp):

        def setUp(self):
            '''Test gaussian_random op with specified value
            '''
            self.init()
            self.init_data()
            self.use_mkldnn = False

            self.inputs = {"ShapeTensor": np.array(self.shape).astype("int32")}
            self.attrs = {
                'mean': self.mean,
                'std': self.std,
                'seed': self.seed,
                'use_mkldnn': self.use_mkldnn
            }
            self.outputs = {'Out': np.zeros((123, 92), dtype=self.dtype)}

        def init_data(self):
            self.shape = [123, 92]
            self.use_mkldnn = False
            self.mean = 1.0
            self.std = 2.0
            self.seed = 10


# Test python API
class TestGaussianRandomAPI(unittest.TestCase):

    def test_api(self):
        positive_2_int32 = fluid.layers.fill_constant([1], "int32", 2000)

        positive_2_int64 = fluid.layers.fill_constant([1], "int64", 500)
        shape_tensor_int32 = fluid.data(name="shape_tensor_int32",
                                        shape=[2],
                                        dtype="int32")

        shape_tensor_int64 = fluid.data(name="shape_tensor_int64",
                                        shape=[2],
                                        dtype="int64")

        out_1 = fluid.layers.gaussian_random(shape=[2000, 500],
                                             dtype="float32",
                                             mean=0.0,
                                             std=1.0,
                                             seed=10)

        out_2 = fluid.layers.gaussian_random(shape=[2000, positive_2_int32],
                                             dtype="float32",
                                             mean=0.,
                                             std=1.0,
                                             seed=10)

        out_3 = fluid.layers.gaussian_random(shape=[2000, positive_2_int64],
                                             dtype="float32",
                                             mean=0.,
                                             std=1.0,
                                             seed=10)

        out_4 = fluid.layers.gaussian_random(shape=shape_tensor_int32,
                                             dtype="float32",
                                             mean=0.,
                                             std=1.0,
                                             seed=10)

        out_5 = fluid.layers.gaussian_random(shape=shape_tensor_int64,
                                             dtype="float32",
                                             mean=0.,
                                             std=1.0,
                                             seed=10)

        out_6 = fluid.layers.gaussian_random(shape=shape_tensor_int64,
                                             dtype=np.float32,
                                             mean=0.,
                                             std=1.0,
                                             seed=10)

        exe = fluid.Executor(place=fluid.XPUPlace(0))
        res_1, res_2, res_3, res_4, res_5, res_6 = exe.run(
            fluid.default_main_program(),
            feed={
                "shape_tensor_int32": np.array([2000, 500]).astype("int32"),
                "shape_tensor_int64": np.array([2000, 500]).astype("int64"),
            },
            fetch_list=[out_1, out_2, out_3, out_4, out_5, out_6])

        self.assertAlmostEqual(np.mean(res_1), 0.0, delta=0.1)
        self.assertAlmostEqual(np.std(res_1), 1., delta=0.1)
        self.assertAlmostEqual(np.mean(res_2), 0.0, delta=0.1)
        self.assertAlmostEqual(np.std(res_2), 1., delta=0.1)
        self.assertAlmostEqual(np.mean(res_3), 0.0, delta=0.1)
        self.assertAlmostEqual(np.std(res_3), 1., delta=0.1)
        self.assertAlmostEqual(np.mean(res_4), 0.0, delta=0.1)
        self.assertAlmostEqual(np.std(res_5), 1., delta=0.1)
        self.assertAlmostEqual(np.mean(res_5), 0.0, delta=0.1)
        self.assertAlmostEqual(np.std(res_5), 1., delta=0.1)
        self.assertAlmostEqual(np.mean(res_6), 0.0, delta=0.1)
        self.assertAlmostEqual(np.std(res_6), 1., delta=0.1)

    def test_default_dtype(self):
        paddle.disable_static()

        def test_default_fp16():
            paddle.framework.set_default_dtype('float16')
            paddle.tensor.random.gaussian([2, 3])

        self.assertRaises(TypeError, test_default_fp16)

        def test_default_fp32():
            paddle.framework.set_default_dtype('float32')
            out = paddle.tensor.random.gaussian([2, 3])
            self.assertEqual(out.dtype, fluid.core.VarDesc.VarType.FP32)

        def test_default_fp64():
            paddle.framework.set_default_dtype('float64')
            out = paddle.tensor.random.gaussian([2, 3])
            self.assertEqual(out.dtype, fluid.core.VarDesc.VarType.FP64)

        test_default_fp64()
        test_default_fp32()

        paddle.enable_static()


class TestStandardNormalDtype(unittest.TestCase):

    def test_default_dtype(self):
        paddle.disable_static()

        def test_default_fp16():
            paddle.framework.set_default_dtype('float16')
            paddle.tensor.random.standard_normal([2, 3])

        self.assertRaises(TypeError, test_default_fp16)

        def test_default_fp32():
            paddle.framework.set_default_dtype('float32')
            out = paddle.tensor.random.standard_normal([2, 3])
            self.assertEqual(out.dtype, fluid.core.VarDesc.VarType.FP32)

        def test_default_fp64():
            paddle.framework.set_default_dtype('float64')
            out = paddle.tensor.random.standard_normal([2, 3])
            self.assertEqual(out.dtype, fluid.core.VarDesc.VarType.FP64)

        test_default_fp64()
        test_default_fp32()

        paddle.enable_static()


support_types = get_xpu_op_support_types('gaussian_random')
for stype in support_types:
    create_test_class(globals(), XPUTestGaussianRandomOp, stype)

if __name__ == "__main__":
    unittest.main()
