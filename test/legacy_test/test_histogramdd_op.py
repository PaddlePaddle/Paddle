# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# inputs, bins, range, weights, density
class TestHistogramddAPI(unittest.TestCase):
    def setUp(self):
        self.range = None
        self.weights = None
        self.density = False

        self.init_input()
        self.set_expect_output()
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def init_input(self):
        self.sample = np.array([[0., 1.], [1., 0.], [2., 0.], [2., 2.]])
        self.bins = [3, 3]
        self.weights = np.array([1., 2., 4., 8.])
    
    def set_expect_output(self):  # pytorch output as expect_hist/expect_edges
        self.expect_hist = np.array([[0., 1., 0.],
                                     [2., 0., 0.],
                                     [4., 0., 8.]])
        self.expect_edges = [[0.0000, 0.6667, 1.3333, 2.0000],
                             [0.0000, 0.6667, 1.3333, 2.0000]]


    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('x', self.sample.shape, dtype=self.sample.dtype)
            if self.weights is not None:
                weights = paddle.static.data('weights', self.weights.shape, dtype=self.weights.dtype)
                out_0, out_1 = paddle.histogramdd(x, bins=self.bins, weights=weights, range=self.range, density=self.density)
            else:
                out_0, out_1 = paddle.histogramdd(x, bins=self.bins, range=self.range, density=self.density)
            exe = paddle.static.Executor(self.place)
            if self.weights is not None:
                res = exe.run(feed={'x': self.sample, 'weights': self.weights}, fetch_list=[out_0, out_1])
            else:
                res = exe.run(feed={'x': self.sample}, fetch_list=[out_0, out_1])

            hist_out, edges_out = res[0], res[1:]

            np.testing.assert_allclose(self.expect_hist, hist_out, rtol=1e-4, atol=1e-4)
            for idx, edge_out in enumerate(edges_out):
                edge_out = edge_out
                expect_edge = np.array(self.expect_edges[idx])
                np.testing.assert_allclose(expect_edge, edge_out, rtol=1e-4, atol=1e-4)


    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        self.sample_dy = paddle.to_tensor(self.sample)
        self.weights_dy = None
        if self.weights is not None:
            self.weights_dy = paddle.to_tensor(self.weights)
        hist, edges = paddle.histogramdd(self.sample_dy, bins=self.bins, weights=self.weights_dy, range=self.range, density=self.density)

        np.testing.assert_allclose(self.expect_hist, hist.numpy(), rtol=1e-4, atol=1e-4)
        for idx, edge in enumerate(edges):
            edge = edge.numpy()
            expect_edge = np.array(self.expect_edges[idx])
            np.testing.assert_allclose(expect_edge, edge, rtol=1e-4, atol=1e-4)

        paddle.enable_static()

class TestHistogramddAPICase1ForDensity(TestHistogramddAPI):
    def init_input(self):
        self.sample = np.array([[0., 0.], [1., 1.], [2., 2.]])
        self.bins = [2, 2]
        self.range = [0., 1., 0., 1.]
        self.density = True

    def set_expect_output(self):
        self.expect_hist = np.array([[2., 0.],
                                     [0., 2.]])
        self.expect_edges = [[0.0000, 0.5000, 1.0000],
                             [0.0000, 0.5000, 1.0000]]

class TestHistogramddAPICase2ForMultiDimsAndDensity(TestHistogramddAPI):
    def init_input(self):
        # shape: [4,2,2]
        self.sample = np.array([
                [[1., 2.], [3., 4.]],
                [[5., 6.], [7., 8.]],
                [[9., 10.], [11., 12.]],
                [[13., 14.], [15., 16.]],
            ])
        self.bins = [3, 4]
        self.density = True

    def set_expect_output(self):
        self.expect_hist = np.array([
                [0.0153, 0.0077, 0.0000, 0.0000],
                [0.0000, 0.0077, 0.0077, 0.0000],
                [0.0000, 0.0000, 0.0077, 0.0153]
            ])
        self.expect_edges = [[1.0000, 5.6667, 10.3333, 15.0000],
                            [2.0000, 5.5000, 9.0000, 12.5000, 16.0000]]



class TestHistogramddAPICase3ForMultiDimsNotDensity(TestHistogramddAPI):
    def init_input(self):
        # shape: [4,2,2]
        self.sample = np.array([
                [[1., 2.], [3., 4.]],
                [[5., 6.], [7., 8.]],
                [[9., 10.], [11., 12.]],
                [[13., 14.], [15., 16.]],
            ])
        self.bins = [3, 4]
        # self.density = True

    def set_expect_output(self):
        self.expect_hist = np.array([
                [2., 1., 0., 0.],
                [0., 1., 1., 0.],
                [0., 0., 1., 2.]
            ])
        self.expect_edges = [[1.0000, 5.6667, 10.3333, 15.0000],
                            [2.0000, 5.5000, 9.0000, 12.5000, 16.0000]]



class TestHistogramddAPICase4ForRangeAndDensity(TestHistogramddAPI):
    def init_input(self):
        # shape: [4,2,2]
        self.sample = np.array([
                [[1., 2.], [3., 4.]],
                [[5., 6.], [7., 8.]],
                [[9., 10.], [11., 12.]],
                [[13., 14.], [15., 16.]],
            ])
        self.bins = [3, 4]
        # [leftmost_1, rightmost_1, leftmost_2, rightmost_2,..., leftmost_D, rightmost_D]
        self.range = [1., 10., 1., 100.]
        self.density = True

    def set_expect_output(self):
        self.expect_hist = np.array([
                [0.0054, 0.0000, 0.0000, 0.0000],
                [0.0027, 0.0000, 0.0000, 0.0000],
                [0.0054, 0.0000, 0.0000, 0.0000]
            ])
        self.expect_edges = [[1., 4., 7. ,10.],
                            [1.0000, 25.7500, 50.5000, 75.2500 ,100.0000]]




class TestHistogramddAPICase5ForRangeNotDensity(TestHistogramddAPI):
    def init_input(self):
        # shape: [4,2,2]
        self.sample = np.array([
                [[1., 2.], [3., 4.]],
                [[5., 6.], [7., 8.]],
                [[9., 10.], [11., 12.]],
                [[13., 14.], [15., 16.]],
            ])
        self.bins = [3, 4]
        # [leftmost_1, rightmost_1, leftmost_2, rightmost_2,..., leftmost_D, rightmost_D]
        self.range = [1., 10., 1., 100.]
        # self.density = True

    def set_expect_output(self):
        self.expect_hist = np.array([
                [2., 0., 0., 0.],
                [1., 0., 0., 0.],
                [2., 0., 0., 0.]
            ])
        self.expect_edges = [[1., 4., 7. ,10.],
                            [1.0000, 25.7500, 50.5000, 75.2500 ,100.0000]]



class TestHistogramddAPICase6NotRangeAndDensityAndWeights(TestHistogramddAPI):
    def init_input(self):
        # shape: [4,2,2]
        self.sample = np.array([
                [[1., 2.], [3., 4.]],
                [[5., 6.], [7., 8.]],
                [[9., 10.], [11., 12.]],
                [[13., 14.], [15., 16.]],
            ])
        self.bins = [3, 4]
        # [leftmost_1, rightmost_1, leftmost_2, rightmost_2,..., leftmost_D, rightmost_D]
        # self.range = [1., 10., 1., 100.]
        self.density = True
        self.weights = np.array([
            [1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.],
        ])

    def set_expect_output(self):
        self.expect_hist = np.array([[0.0051, 0.0051, 0.0000, 0.0000],
                                    [0.0000, 0.0068, 0.0085, 0.0000],
                                    [0.0000, 0.0000, 0.0102, 0.0255]])

        self.expect_edges = [[ 1.0000,  5.6667, 10.3333, 15.0000],
                            [ 2.0000,  5.5000,  9.0000, 12.5000, 16.0000]]


class TestHistogramddAPICase7ForRangeAndDensityAndWeights(TestHistogramddAPI):
    def init_input(self):
        # shape: [4,2,2]
        self.sample = np.array([
                [[1., 2.], [3., 4.]],
                [[5., 6.], [7., 8.]],
                [[9., 10.], [11., 12.]],
                [[13., 14.], [15., 16.]],
            ])
        self.bins = [3, 4]
        # [leftmost_1, rightmost_1, leftmost_2, rightmost_2,..., leftmost_D, rightmost_D]
        self.range = [1., 10., 1., 100.]
        self.density = True
        self.weights = np.array([
            [1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.],
        ])

    def set_expect_output(self):
        self.expect_hist = np.array([[0.0027, 0.0000, 0.0000, 0.0000],
                                    [0.0027, 0.0000, 0.0000, 0.0000],
                                    [0.0081, 0.0000, 0.0000, 0.0000]])
        self.expect_edges = [[1., 4., 7. ,10.],
                            [1.0000, 25.7500, 50.5000, 75.2500 ,100.0000]]


class TestHistogramddAPICase8MoreInnermostDim(TestHistogramddAPI):
    def init_input(self):
        # shape: [4,2,4]
        self.sample = np.array([
                [[1., 2., 6., 12.], [3., 4., 7, 12.]],
                [[5., 6., 3., 8.], [7., 8., 1., 7.]],
                [[9., 10., 9., 4.], [11., 12., 8., 5.]],
                [[13., 14., 10., 9.], [15., 16., 5., 2.]],
            ])
        self.bins = [1, 2, 3, 4]
        # [leftmost_1, rightmost_1, leftmost_2, rightmost_2,..., leftmost_D, rightmost_D]
        self.density = False
        self.weights = np.array([
            [1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.],
        ])

    def set_expect_output(self):
        self.expect_hist = np.array([[[[0., 0., 7., 0.],
                                       [0., 0., 0., 1.],
                                       [0., 0., 0., 2.]],

                                      [[0., 0., 0., 0.],
                                       [8., 0., 0., 0.],
                                       [5., 6., 7., 0.]]]])
        self.expect_edges = [[ 1., 15.],
                            [ 2.,  9., 16.],
                            [ 1.,  4.,  7., 10.],
                            [ 2.0000,  4.5000,  7.0000,  9.5000, 12.0000]]

class TestHistogramddAPICase8MoreInnermostDimAndDensity(TestHistogramddAPI):
    def init_input(self):
        # shape: [4,2,4]
        self.sample = np.array([
                [[1., 2., 6., 12.], [3., 4., 7, 12.]],
                [[5., 6., 3., 8.], [7., 8., 1., 7.]],
                [[9., 10., 9., 4.], [11., 12., 8., 5.]],
                [[13., 14., 10., 9.], [15., 16., 5., 2.]],
            ])
        self.bins = [1, 2, 3, 4]
        # [leftmost_1, rightmost_1, leftmost_2, rightmost_2,..., leftmost_D, rightmost_D]
        self.density = True
        self.weights = np.array([
            [1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.],
        ])

    def set_expect_output(self):
        self.expect_hist = np.array([[[[0.0000e+00, 0.0000e+00, 2.6455e-04, 0.0000e+00],
                                       [0.0000e+00, 0.0000e+00, 0.0000e+00, 3.7793e-05],
                                       [0.0000e+00, 0.0000e+00, 0.0000e+00, 7.5586e-05]],

                                      [[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                                       [3.0234e-04, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                                       [1.8896e-04, 2.2676e-04, 2.6455e-04, 0.0000e+00]]]])
        self.expect_edges = [[ 1., 15.],
                            [ 2.,  9., 16.],
                            [ 1.,  4.,  7., 10.],
                            [ 2.0000,  4.5000,  7.0000,  9.5000, 12.0000]]

if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
