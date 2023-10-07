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
        self.expect_edges = [[0.00000000, 0.66666669, 1.33333325, 2.00000000],
                             [0.00000000, 0.66666669, 1.33333325, 2.00000000]]


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

            np.testing.assert_allclose(hist_out, self.expect_hist, atol=1e-8)
            for idx, edge_out in enumerate(edges_out):
                edge_out = edge_out
                expect_edge = np.array(self.expect_edges[idx])
                np.testing.assert_allclose(edge_out, expect_edge, atol=1e-8)


    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        self.sample_dy = paddle.to_tensor(self.sample)
        self.weights_dy = None
        if self.weights is not None:
            self.weights_dy = paddle.to_tensor(self.weights)
        if isinstance(self.bins, tuple):
            self.bins = tuple([paddle.to_tensor(bin) for bin in self.bins])
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
        self.expect_edges = [[0.00000000, 0.50000000, 1.00000000],
                             [0.00000000, 0.50000000, 1.00000000]]

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
                [0.01530612, 0.00765306, 0.00000000, 0.00000000],
                [0.00000000, 0.00765306, 0.00765306, 0.00000000],
                [0.00000000, 0.00000000, 0.00765306, 0.01530612]
            ])
        self.expect_edges = [[ 1.00000000,  5.66666651, 10.33333397, 15.00000000],
                            [ 2.00000000,  5.50000000,  9.00000000, 12.50000000, 16.00000000]]



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
        self.expect_edges = [[ 1.00000000,  5.66666651, 10.33333397, 15.00000000],
                            [ 2.00000000,  5.50000000,  9.00000000, 12.50000000, 16.00000000]]



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
                [0.00538721, 0.00000000, 0.00000000, 0.00000000],
                [0.00269360, 0.00000000, 0.00000000, 0.00000000],
                [0.00538721, 0.00000000, 0.00000000, 0.00000000]
            ])
        self.expect_edges = [[1., 4., 7. ,10.],
                            [  1.00000000,  25.75000000,  50.50000000,  75.25000000, 100.00000000]]




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
                            [  1.00000000,  25.75000000,  50.50000000,  75.25000000, 100.00000000]]



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
        self.expect_hist = np.array([[0.00510204, 0.00510204, 0.00000000, 0.00000000],
                                    [0.00000000, 0.00680272, 0.00850340, 0.00000000],
                                    [0.00000000, 0.00000000, 0.01020408, 0.02551021]])

        self.expect_edges = [[ 1.00000000,  5.66666651, 10.33333397, 15.00000000],
                            [ 2.00000000,  5.50000000,  9.00000000, 12.50000000, 16.00000000]]


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
        self.expect_hist = np.array([[0.00269360, 0.00000000, 0.00000000, 0.00000000],
                                    [0.00269360, 0.00000000, 0.00000000, 0.00000000],
                                    [0.00808081, 0.00000000, 0.00000000, 0.00000000]])
        self.expect_edges = [[1., 4., 7. ,10.],
                            [1.00000000, 25.75000000, 50.50000000, 75.25000000 ,100.00000000]]


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
                            [ 2.00000000,  4.50000000,  7.00000000,  9.50000000, 12.00000000]]

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
        self.expect_hist = np.array([[[[0.00000000e+00, 0.00000000e+00, 2.64550268e-04, 0.00000000e+00],
                                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.77928955e-05],
                                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 7.55857909e-05]],

                                      [[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                       [3.02343164e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                       [1.88964477e-04, 2.26757373e-04, 2.64550268e-04, 0.00000000e+00]]]])
        self.expect_edges = [[ 1., 15.],
                            [ 2.,  9., 16.],
                            [ 1.,  4.,  7., 10.],
                            [ 2.00000000,  4.50000000,  7.00000000,  9.50000000, 12.00000000]]
        
class TestHistogramddAPICase9ForIntBin(TestHistogramddAPI):
    def init_input(self):
        # shape: [4,2,2]
        self.sample = np.array([
                [[1., 2.], [3., 4.]],
                [[5., 6.], [7., 8.]],
                [[9., 10.], [11., 12.]],
                [[13., 14.], [15., 16.]],
            ])
        self.weights = np.array([
            [1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.],
        ])
        self.bins = 5
        self.density = True
        self.range = [1., 10., 1., 100.]


    def set_expect_output(self):
        self.expect_hist = np.array([[0.00187056, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                                    [0.00374112, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                                    [0.00561167, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                                    [0.00748223, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                                    [0.00935279, 0.00000000, 0.00000000, 0.00000000, 0.00000000]])
        self.expect_edges = [[ 1.00000000,  2.79999995,  4.59999990,  6.40000010,  8.19999981, 10.00000000],
                            [  1.00000000,  20.79999924,  40.59999847,  60.40000153,  80.19999695, 100.00000000]]

class TestHistogramddAPICase10ForTensorBin(TestHistogramddAPI):
    def init_input(self):
        # shape: [4,2,2]
        self.sample = np.array([
                [[1., 2.], [3., 4.]],
                [[5., 6.], [7., 8.]],
                [[9., 10.], [11., 12.]],
                [[13., 14.], [15., 16.]],
            ])
        self.weights = np.array([
            [1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.],
        ])
        self.bins = (np.array([1., 2., 10., 15., 20.]), np.array([0., 20., 100.]))
        self.density = True
        

    def set_expect_output(self):
        self.expect_hist = np.array([[0.00138889, 0.00000000],
                                    [0.00243056, 0.00000000],
                                    [0.00361111, 0.00000000],
                                    [0.00222222, 0.00000000]])
        self.expect_edges = [[ 1.,  2., 10., 15., 20.],
                            [  0.,  20., 100.]]



if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
