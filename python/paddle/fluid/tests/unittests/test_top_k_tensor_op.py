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

import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid.core as core


def numpy_topk_list(x, k=[1], axis=-1, largest=True):
    if axis < 0:
        axis = len(x.shape) + axis

    last_dim = len(x.shape) - 1
    need_transpose = False
    if axis != last_dim:
        need_transpose = True

    if need_transpose:
        dims = np.arange(len(x.shape))
        tmp = dims[-1]
        dims[-1] = axis
        dims[axis] = tmp
        x = x.transpose(dims)

    if largest:
        indices = np.argsort(-x, axis=-1)
    else:
        indices = np.argsort(x, axis=-1)
    if largest:
        value = -np.sort(-x, axis=-1)
    else:
        value = np.sort(x, axis=-1)

    k_largest = np.max(k)
    indices_res = indices.take(indices=range(0, k_largest), axis=-1)
    value_res = value.take(indices=range(0, k_largest), axis=-1)
    threshold = value_res.copy()

    for i in range(x.shape[0]):
        threshold[i][:] = value_res[i].take(indices=range(0, k[i]),
                                            axis=-1)[..., -1:]

    if largest:
        indices_res = np.where(value_res >= threshold, indices_res, 0)
        value_res = np.where(value_res >= threshold, value_res, 0)
    else:
        indices_res = np.where(value_res <= threshold, indices_res, 0)
        value_res = np.where(value_res <= threshold, value_res, 0)

    if need_transpose:
        value_res = value_res.transpose(dims)
        indices_res = indices_res.transpose(dims)

    return value_res, indices_res


class TestTopKTensorOp(OpTest):

    def init_args(self):
        self.k = np.array([1, 2, 3, 4, 5]).astype(np.int32)
        self.axis = -1
        self.largest = True

    def _get_places(self):
        places = []
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        return places

    def setUp(self):
        paddle.enable_static()
        self.op_type = "top_k_tensor"
        self.python_api = paddle.top_k_tensor
        self.dtype = np.float64
        self.input_data = np.random.rand(5, 20)
        self.init_args()
        self.inputs = {'x': self.input_data, 'k_list': self.k}
        self.attrs = {'axis': self.axis, 'largest': self.largest}
        output, indices = numpy_topk_list(x=self.input_data,
                                          k=self.k,
                                          axis=self.axis,
                                          largest=self.largest)
        self.outputs = {'out': output, 'indices': indices}

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad(self):
        self.check_grad(['x'], 'out', check_eager=True)


class TestTopKTensorOp2(OpTest):

    def init_args(self):
        self.k = np.array([1, 2, 3, 4, 5]).astype(np.int32)
        self.axis = 1
        self.largest = True

    def setUp(self):
        self.op_type = "top_k_tensor"
        self.python_api = paddle.top_k_tensor
        self.dtype = np.float64
        self.input_data = np.random.rand(5, 10, 5)
        self.init_args()
        self.inputs = {'x': self.input_data, 'k_list': self.k}
        self.attrs = {'axis': self.axis, 'largest': self.largest}
        output, indices = numpy_topk_list(x=self.input_data,
                                          k=self.k,
                                          axis=self.axis,
                                          largest=self.largest)
        self.outputs = {'out': output, 'indices': indices}


class TestTopKTensorAPI(unittest.TestCase):

    def setUp(self):
        np.random.seed(123)
        self.input_data = np.random.rand(6, 7, 8)
        self.k_np = np.array([1, 2, 3, 4, 5, 6]).astype("int32")

    def run_dygraph(self, place):
        with paddle.fluid.dygraph.guard(place):
            input_tensor = paddle.to_tensor(self.input_data)
            k_tensor = paddle.to_tensor(self.k_np)
            # test case for basic test case 1
            paddle_result = paddle.top_k_tensor(input_tensor, k=k_tensor)
            numpy_result = numpy_topk_list(self.input_data, k=self.k_np)
            np.testing.assert_allclose(paddle_result[0].numpy(),
                                       numpy_result[0],
                                       rtol=1e-05)
            np.testing.assert_allclose(paddle_result[1].numpy(),
                                       numpy_result[1],
                                       rtol=1e-05)
            # test case for basic test case 2 with axis
            paddle_result = paddle.top_k_tensor(input_tensor,
                                                k=k_tensor,
                                                axis=1)
            numpy_result = numpy_topk_list(self.input_data, k=self.k_np, axis=1)
            np.testing.assert_allclose(paddle_result[0].numpy(),
                                       numpy_result[0],
                                       rtol=1e-05)
            np.testing.assert_allclose(paddle_result[1].numpy(),
                                       numpy_result[1],
                                       rtol=1e-05)
            # test case for basic test case 3 with tensor largest
            paddle_result = paddle.top_k_tensor(input_tensor,
                                                k=k_tensor,
                                                axis=1,
                                                largest=False)
            numpy_result = numpy_topk_list(self.input_data,
                                           k=self.k_np,
                                           axis=1,
                                           largest=False)
            np.testing.assert_allclose(paddle_result[0].numpy(),
                                       numpy_result[0],
                                       rtol=1e-05)
            np.testing.assert_allclose(paddle_result[1].numpy(),
                                       numpy_result[1],
                                       rtol=1e-05)

    def run_static(self, place):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            input_tensor = paddle.static.data(name="x",
                                              shape=[6, 7, 8],
                                              dtype="float64")
            k_tensor = paddle.static.data(name="k", shape=[6], dtype="int32")
            result1 = paddle.top_k_tensor(input_tensor, k=k_tensor)
            result2 = paddle.top_k_tensor(input_tensor, k=k_tensor, axis=1)
            self.assertEqual(result2[0].shape, (6, -1, 8))
            self.assertEqual(result2[1].shape, (6, -1, 8))
            exe = paddle.static.Executor(place)
            paddle_result = exe.run(
                feed={
                    "x": self.input_data,
                    "k": self.k_np
                },
                fetch_list=[result1[0], result1[1], result2[0], result2[1]])
            numpy_result = numpy_topk_list(self.input_data, k=self.k_np)
            np.testing.assert_allclose(paddle_result[0],
                                       numpy_result[0],
                                       rtol=1e-05,
                                       atol=1e-05)
            np.testing.assert_allclose(paddle_result[1],
                                       numpy_result[1],
                                       rtol=1e-05,
                                       atol=1e-05)
            numpy_result = numpy_topk_list(self.input_data, k=self.k_np, axis=1)
            np.testing.assert_allclose(paddle_result[2],
                                       numpy_result[0],
                                       rtol=1e-05,
                                       atol=1e-05)
            np.testing.assert_allclose(paddle_result[3],
                                       numpy_result[1],
                                       rtol=1e-05,
                                       atol=1e-05)

    def test_cases(self):
        places = []
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.run_dygraph(place)
            paddle.device.cuda.empty_cache()
            self.run_static(place)


if __name__ == "__main__":
    unittest.main()
