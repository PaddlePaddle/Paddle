# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid
import numpy as np
import unittest


def np_pairwise_distance(x, y, p=2.0, epsilon=1e-6, keepdim=False):
    distance = np.linalg.norm(x - y + epsilon, ord=p, axis=-1, keepdims=keepdim)
    # Paddle currently has not supported for 0-d Tensors, so even if keep_dim is False,
    # and neither x nor y is batched, a Tensor of shape (1, ) is returned
    if distance.ndim == 0:
        distance = np.expand_dims(distance, axis=0)
    return distance


def call_pairwise_distance_layer(x, y, p=2., epsilon=1e-6, keepdim='False'):
    pairwise_distance = paddle.nn.PairwiseDistance(p=p,
                                                   epsilon=epsilon,
                                                   keepdim=keepdim)
    distance = pairwise_distance(x=x, y=y)
    return distance


def call_pairwise_distance_functional(x,
                                      y,
                                      p=2.,
                                      epsilon=1e-6,
                                      keepdim='False'):
    distance = paddle.nn.functional.pairwise_distance(x=x,
                                                      y=y,
                                                      p=p,
                                                      epsilon=epsilon,
                                                      keepdim=keepdim)
    return distance


def test_static(place,
                x_np,
                y_np,
                p=2.0,
                epsilon=1e-6,
                keepdim=False,
                functional=False):
    prog = paddle.static.Program()
    startup_prog = paddle.static.Program()
    place = fluid.CUDAPlace(
        0) if paddle.fluid.core.is_compiled_with_cuda() else fluid.CPUPlace()
    paddle.enable_static()
    with paddle.static.program_guard(prog, startup_prog):
        x = paddle.fluid.data(name='x', shape=x_np.shape, dtype=x_np.dtype)
        y = paddle.fluid.data(name='y', shape=y_np.shape, dtype=x_np.dtype)

        if functional:
            distance = call_pairwise_distance_functional(x=x,
                                                         y=y,
                                                         p=p,
                                                         epsilon=epsilon,
                                                         keepdim=keepdim)
        else:
            distance = call_pairwise_distance_layer(x=x,
                                                    y=y,
                                                    p=p,
                                                    epsilon=epsilon,
                                                    keepdim=keepdim)
        exe = paddle.static.Executor(place)
        static_ret = exe.run(prog,
                             feed={
                                 'x': x_np,
                                 'y': y_np
                             },
                             fetch_list=[distance])
        static_ret = static_ret[0]
    paddle.disable_static()
    return static_ret


def test_dygraph(place,
                 x_np,
                 y_np,
                 p=2.0,
                 epsilon=1e-6,
                 keepdim=False,
                 functional=False):
    x = paddle.to_tensor(x_np)
    y = paddle.to_tensor(y_np)
    if functional:
        dy_distance = call_pairwise_distance_functional(x=x,
                                                        y=y,
                                                        p=p,
                                                        epsilon=epsilon,
                                                        keepdim=keepdim)
    else:
        dy_distance = call_pairwise_distance_layer(x=x,
                                                   y=y,
                                                   p=p,
                                                   epsilon=epsilon,
                                                   keepdim=keepdim)
    dygraph_ret = dy_distance.numpy()
    return dygraph_ret


def test_legacy_dygraph(place,
                        x_np,
                        y_np,
                        p=2.0,
                        epsilon=1e-6,
                        keepdim=False,
                        functional=False):
    paddle.fluid.framework._enable_legacy_dygraph()
    x = paddle.to_tensor(x_np)
    y = paddle.to_tensor(y_np)
    if functional:
        legacy_distance = call_pairwise_distance_functional(x=x,
                                                            y=y,
                                                            p=p,
                                                            epsilon=epsilon,
                                                            keepdim=keepdim)
    else:
        legacy_distance = call_pairwise_distance_layer(x=x,
                                                       y=y,
                                                       p=p,
                                                       epsilon=epsilon,
                                                       keepdim=keepdim)
    legacy_ret = legacy_distance.numpy()
    paddle.fluid.framework._disable_legacy_dygraph()
    return legacy_ret


class TestPairwiseDistance(unittest.TestCase):

    def test_pairwise_distance(self):
        epsilon = 1e-6
        all_shape = [[5], [100, 100]]
        dtypes = ['float32', 'float64']
        p_list = [-1, 0, 1, 2, np.inf, -np.inf]
        places = [paddle.CPUPlace()]
        if paddle.device.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        keeps = [False, True]
        for place in places:
            for shape in all_shape:
                for dtype in dtypes:
                    for p in p_list:
                        for keepdim in keeps:
                            x_np = np.random.random(shape).astype(dtype)
                            y_np = np.random.random(shape).astype(dtype)

                            static_ret = test_static(place,
                                                     x_np,
                                                     y_np,
                                                     p,
                                                     epsilon=epsilon,
                                                     keepdim=keepdim)
                            dygraph_ret = test_dygraph(place,
                                                       x_np,
                                                       y_np,
                                                       p,
                                                       epsilon=epsilon,
                                                       keepdim=keepdim)
                            legacy_ret = test_legacy_dygraph(place,
                                                             x_np,
                                                             y_np,
                                                             p,
                                                             epsilon=epsilon,
                                                             keepdim=keepdim)
                            excepted_value = np_pairwise_distance(
                                x_np, y_np, p, epsilon=epsilon, keepdim=keepdim)

                            self.assertEqual(static_ret.shape,
                                             excepted_value.shape)
                            self.assertEqual(dygraph_ret.shape,
                                             excepted_value.shape)
                            self.assertEqual(legacy_ret.shape,
                                             excepted_value.shape)

                            np.testing.assert_allclose(static_ret,
                                                       excepted_value,
                                                       rtol=1e-05)
                            np.testing.assert_allclose(dygraph_ret,
                                                       excepted_value,
                                                       rtol=1e-05)
                            np.testing.assert_allclose(legacy_ret,
                                                       excepted_value,
                                                       rtol=1e-05)

                            static_functional_ret = test_static(place,
                                                                x_np,
                                                                y_np,
                                                                p,
                                                                epsilon=epsilon,
                                                                keepdim=keepdim)
                            dygraph_functional_ret = test_dygraph(
                                place,
                                x_np,
                                y_np,
                                p,
                                epsilon=epsilon,
                                keepdim=keepdim)
                            legacy_functional_ret = test_legacy_dygraph(
                                place,
                                x_np,
                                y_np,
                                p,
                                epsilon=epsilon,
                                keepdim=keepdim)

                            self.assertEqual(static_functional_ret.shape,
                                             excepted_value.shape)
                            self.assertEqual(dygraph_functional_ret.shape,
                                             excepted_value.shape)
                            self.assertEqual(legacy_functional_ret.shape,
                                             excepted_value.shape)

                            np.testing.assert_allclose(static_functional_ret,
                                                       excepted_value,
                                                       rtol=1e-05)
                            np.testing.assert_allclose(dygraph_functional_ret,
                                                       excepted_value,
                                                       rtol=1e-05)
                            np.testing.assert_allclose(legacy_functional_ret,
                                                       excepted_value,
                                                       rtol=1e-05)

    def test_pairwise_distance_broadcast_1(self):
        shape_x = [100, 100]
        shape_y = [100, 1]
        epsilon = 1e-6
        keepdim = False
        place = paddle.CPUPlace()
        x_np = np.random.random(shape_x).astype('float32')
        y_np = np.random.random(shape_y).astype('float32')
        static_ret = test_static(place=place,
                                 x_np=x_np,
                                 y_np=y_np,
                                 epsilon=epsilon,
                                 keepdim=keepdim)
        dygraph_ret = test_dygraph(place=place,
                                   x_np=x_np,
                                   y_np=y_np,
                                   epsilon=epsilon,
                                   keepdim=keepdim)
        legacy_ret = test_legacy_dygraph(place=place,
                                         x_np=x_np,
                                         y_np=y_np,
                                         epsilon=epsilon,
                                         keepdim=keepdim)
        excepted_value = np_pairwise_distance(x_np,
                                              y_np,
                                              epsilon=epsilon,
                                              keepdim=keepdim)

        self.assertEqual(static_ret.shape, excepted_value.shape)
        self.assertEqual(dygraph_ret.shape, excepted_value.shape)
        self.assertEqual(legacy_ret.shape, excepted_value.shape)

        np.testing.assert_allclose(static_ret, excepted_value, rtol=1e-05)
        np.testing.assert_allclose(dygraph_ret, excepted_value, rtol=1e-05)
        np.testing.assert_allclose(legacy_ret, excepted_value, rtol=1e-05)

        static_functional_ret = test_static(place=place,
                                            x_np=x_np,
                                            y_np=y_np,
                                            epsilon=epsilon,
                                            keepdim=keepdim,
                                            functional=True)
        dygraph_functional_ret = test_dygraph(place=place,
                                              x_np=x_np,
                                              y_np=y_np,
                                              epsilon=epsilon,
                                              keepdim=keepdim,
                                              functional=True)
        legacy_functional_ret = test_legacy_dygraph(place=place,
                                                    x_np=x_np,
                                                    y_np=y_np,
                                                    epsilon=epsilon,
                                                    keepdim=keepdim,
                                                    functional=True)

        self.assertEqual(static_functional_ret.shape, excepted_value.shape)
        self.assertEqual(dygraph_functional_ret.shape, excepted_value.shape)
        self.assertEqual(legacy_functional_ret.shape, excepted_value.shape)

        np.testing.assert_allclose(static_functional_ret,
                                   excepted_value,
                                   rtol=1e-05)
        np.testing.assert_allclose(dygraph_functional_ret,
                                   excepted_value,
                                   rtol=1e-05)
        np.testing.assert_allclose(legacy_functional_ret,
                                   excepted_value,
                                   rtol=1e-05)

    def test_pairwise_distance_broadcast_2(self):
        shape_x = [100, 100]
        shape_y = [100]
        epsilon = 1e-6
        keepdim = False
        place = paddle.CPUPlace()
        x_np = np.random.random(shape_x).astype('float32')
        y_np = np.random.random(shape_y).astype('float32')
        static_ret = test_static(place=place,
                                 x_np=x_np,
                                 y_np=y_np,
                                 epsilon=epsilon,
                                 keepdim=keepdim)
        dygraph_ret = test_dygraph(place=place,
                                   x_np=x_np,
                                   y_np=y_np,
                                   epsilon=epsilon,
                                   keepdim=keepdim)
        legacy_ret = test_legacy_dygraph(place=place,
                                         x_np=x_np,
                                         y_np=y_np,
                                         epsilon=epsilon,
                                         keepdim=keepdim)
        excepted_value = np_pairwise_distance(x_np,
                                              y_np,
                                              epsilon=epsilon,
                                              keepdim=keepdim)

        self.assertEqual(static_ret.shape, excepted_value.shape)
        self.assertEqual(dygraph_ret.shape, excepted_value.shape)
        self.assertEqual(legacy_ret.shape, excepted_value.shape)

        np.testing.assert_allclose(static_ret, excepted_value, rtol=1e-05)
        np.testing.assert_allclose(dygraph_ret, excepted_value, rtol=1e-05)
        np.testing.assert_allclose(legacy_ret, excepted_value, rtol=1e-05)

        static_functional_ret = test_static(place=place,
                                            x_np=x_np,
                                            y_np=y_np,
                                            epsilon=epsilon,
                                            keepdim=keepdim,
                                            functional=True)
        dygraph_functional_ret = test_dygraph(place=place,
                                              x_np=x_np,
                                              y_np=y_np,
                                              epsilon=epsilon,
                                              keepdim=keepdim,
                                              functional=True)
        legacy_functional_ret = test_legacy_dygraph(place=place,
                                                    x_np=x_np,
                                                    y_np=y_np,
                                                    epsilon=epsilon,
                                                    keepdim=keepdim,
                                                    functional=True)

        self.assertEqual(static_functional_ret.shape, excepted_value.shape)
        self.assertEqual(dygraph_functional_ret.shape, excepted_value.shape)
        self.assertEqual(legacy_functional_ret.shape, excepted_value.shape)

        np.testing.assert_allclose(static_functional_ret,
                                   excepted_value,
                                   rtol=1e-05)
        np.testing.assert_allclose(dygraph_functional_ret,
                                   excepted_value,
                                   rtol=1e-05)
        np.testing.assert_allclose(legacy_functional_ret,
                                   excepted_value,
                                   rtol=1e-05)


if __name__ == "__main__":
    unittest.main()
