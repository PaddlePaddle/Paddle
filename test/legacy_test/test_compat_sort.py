# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.compat import sort as compat_sort


class TestCompatSort(unittest.TestCase):
    def _compare_with_origin(
        self, input_tensor, dtype, dim, descending, stable, use_out=False
    ):
        """DO NOT set use_out to be True in static graph mode."""
        if use_out:
            sort_res = (paddle.to_tensor(0), paddle.to_tensor(0))
            compat_sort(input_tensor, dim, descending, stable, out=sort_res)
        else:
            sort_res = compat_sort(
                input_tensor, dim=dim, descending=descending, stable=stable
            )

        origin_vals = paddle.sort(
            input_tensor, axis=dim, descending=descending, stable=stable
        )
        origin_inds = paddle.argsort(
            input_tensor, axis=dim, descending=descending, stable=stable
        )
        if dtype.find("int"):
            np.testing.assert_array_equal(
                sort_res[0].numpy(), origin_vals.numpy()
            )
        else:
            np.testing.assert_allclose(sort_res[0].numpy(), origin_vals.numpy())
        np.testing.assert_array_equal(sort_res[1].numpy(), origin_inds.numpy())

    def test_with_origin_static(self):
        dtypes = [
            "float16",
            "bfloat16",
            "float32",
            "float64",
            "uint8",
            "int16",
            "int32",
            "int64",
        ]
        shapes = [(31, 5), (129,)]
        paddle.seed(1)
        for dtype in dtypes:
            for shape in shapes:
                for dim in range(len(shape)):
                    if dtype.find("int") >= 0:
                        input_tensor = paddle.randint(0, 255, shape).to(dtype)
                    else:
                        input_tensor = paddle.randn(shape, dtype=dtype)

                    def static_graph_tester(descending, stable):
                        with paddle.static.program_guard(
                            paddle.static.Program()
                        ):
                            input_data = paddle.static.data(
                                name='x', shape=shape, dtype=dtype
                            )
                            sort_res = compat_sort(
                                input_data,
                                dim=dim,
                                descending=descending,
                                stable=stable,
                            )
                            sort_vals, sort_inds = (
                                sort_res.values,
                                sort_res.indices,
                            )
                            origin_vals = paddle.sort(
                                input_data,
                                axis=dim,
                                descending=descending,
                                stable=stable,
                            )
                            origin_inds = paddle.argsort(
                                input_data,
                                axis=dim,
                                descending=descending,
                                stable=stable,
                            )
                            place = (
                                paddle.CUDAPlace(0)
                                if paddle.is_compiled_with_cuda()
                                else paddle.CPUPlace()
                            )
                            exe = paddle.static.Executor(place)

                            input_data = np.random.rand(3, 6).astype('float32')
                            feed = {'x': input_tensor.numpy()}
                            results = exe.run(
                                feed=feed,
                                fetch_list=[
                                    sort_vals,
                                    origin_vals,
                                    sort_inds,
                                    origin_inds,
                                ],
                            )
                        if dtype.find("int"):
                            np.testing.assert_array_equal(
                                results[0], results[1]
                            )
                        else:
                            np.testing.assert_allclose(results[0], results[1])
                        np.testing.assert_array_equal(results[2], results[3])

                    paddle.enable_static()
                    static_graph_tester(False, False)
                    static_graph_tester(True, False)
                    static_graph_tester(False, True)
                    static_graph_tester(True, True)
                    paddle.disable_static()

    def test_with_origin_dynamic(self, use_static=False):
        dtypes = [
            "float16",
            "bfloat16",
            "float32",
            "float64",
            "uint8",
            "int16",
            "int32",
            "int64",
        ]
        shapes = [(31, 5), (129,)]
        paddle.seed(0)
        for dtype in dtypes:
            for shape in shapes:
                if dtype.find("int") >= 0:
                    input_tensor = paddle.randint(0, 255, shape).to(dtype)
                else:
                    input_tensor = paddle.randn(shape, dtype=dtype)
                for use_out in [False, True]:
                    for dim in range(len(shape)):
                        self._compare_with_origin(
                            input_tensor,
                            dtype,
                            dim,
                            False,
                            False,
                            use_out=use_out,
                        )
                        self._compare_with_origin(
                            input_tensor,
                            dtype,
                            dim - len(shape),
                            False,
                            True,
                            use_out=use_out,
                        )
                        self._compare_with_origin(
                            input_tensor,
                            dtype,
                            dim,
                            True,
                            False,
                            use_out=use_out,
                        )
                        self._compare_with_origin(
                            input_tensor,
                            dtype,
                            dim - len(shape),
                            True,
                            True,
                            use_out=use_out,
                        )

    def test_sort_backward(self):
        """test the backward behavior for all data types"""
        dtypes = ["float16", "float32", "float64"]
        shapes = [(31, 5), (129,)]
        paddle.seed(2)
        for dtype in dtypes:
            for shape in shapes:
                for dim in range(len(shape)):
                    input_tensor = paddle.randn(shape, dtype=dtype)
                    input_tensor.stop_gradient = False
                    if input_tensor.place.is_gpu_place():
                        y = input_tensor * input_tensor
                    else:
                        y = input_tensor + 1
                    sort_vals, sort_inds = compat_sort(y, dim=dim)
                    sort_vals.backward()
                    if input_tensor.place.is_gpu_place():
                        np.testing.assert_allclose(
                            input_tensor.grad.numpy(),
                            (2 * input_tensor).numpy(),
                        )
                    else:
                        actual_arr = input_tensor.grad.numpy()
                        np.testing.assert_allclose(
                            actual_arr,
                            np.ones_like(actual_arr, dtype=actual_arr.dtype),
                        )

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        x = paddle.to_tensor([])
        sort_res = compat_sort(x, descending=True, stable=True)

        np.testing.assert_array_equal(
            sort_res.values.numpy(), np.array([], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            sort_res.indices.numpy(), np.array([], dtype=np.int64)
        )

        x = paddle.to_tensor(1)
        sort_res = compat_sort(input=x, stable=True)

        np.testing.assert_array_equal(
            sort_res.values.numpy(), np.array(1, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            sort_res.indices.numpy(), np.array(0, dtype=np.int64)
        )

        msg_gt_1 = "paddle.sort() received unexpected keyword arguments 'dim', 'input'. \nDid you mean to use paddle.compat.sort() instead?"
        msg_gt_2 = "paddle.compat.sort() received unexpected keyword arguments 'axis', 'x'. \nDid you mean to use paddle.sort() instead?"

        # invalid split sections
        with self.assertRaises(TypeError) as cm:
            paddle.sort(input=paddle.to_tensor([2, 1, 3]), dim=0)
        self.assertEqual(str(cm.exception), msg_gt_1)

        # invalid split axis
        with self.assertRaises(TypeError) as cm:
            compat_sort(x=paddle.to_tensor([2, 1, 3]), axis=0)
        self.assertEqual(str(cm.exception), msg_gt_2)

        def test_wrong_out_input(dim, out_input):
            with self.assertRaises(TypeError) as cm:
                compat_sort(paddle.to_tensor([1, 2]), out=out_input)

        test_wrong_out_input(0, [0, paddle.to_tensor(0)])
        test_wrong_out_input(0, paddle.to_tensor(0))
        test_wrong_out_input(None, 0)
        test_wrong_out_input(None, (paddle.to_tensor(0),))

        paddle.enable_static()
        with (
            self.assertRaises(RuntimeError) as cm,
            paddle.static.program_guard(paddle.static.Program()),
        ):
            x = paddle.static.data(name='x', shape=[None, 6], dtype='float32')
            result0, result1 = compat_sort(
                paddle.arange(24),
                out=(
                    paddle.zeros([24]),
                    paddle.zeros([24], dtype=paddle.int64),
                ),
            )

            place = (
                paddle.CUDAPlace(0)
                if paddle.is_compiled_with_cuda()
                else paddle.CPUPlace()
            )
            paddle.static.Executor(place).run()
            self.assertEqual(
                str(cm.exception),
                "Using `out` static graph CINN backend is currently not supported. Directly return the tensor tuple instead.\n",
            )
        paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
