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


def ref_histogramdd(x, bins, ranges, weights, density):
    D = x.shape[-1]
    x = x.reshape(-1, D)
    if ranges is not None:
        ranges = np.array(ranges, dtype=x.dtype).reshape(D, 2).tolist()
    if weights is not None:
        weights = weights.reshape(-1)
    ref_hist, ref_edges = np.histogramdd(x, bins, ranges, density, weights)
    return ref_hist, ref_edges


# inputs, bins, ranges, weights, density
class TestHistogramddAPI(unittest.TestCase):
    def setUp(self):
        self.ranges = None
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
        # self.sample = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
        self.sample = np.random.randn(
            4,
            2,
        ).astype(np.float64)
        self.bins = [3, 3]
        self.weights = np.array([1.0, 2.0, 4.0, 8.0], dtype=self.sample.dtype)

    def set_expect_output(self):
        self.expect_hist, self.expect_edges = ref_histogramdd(
            self.sample, self.bins, self.ranges, self.weights, self.density
        )

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(
                'x', self.sample.shape, dtype=self.sample.dtype
            )
            if self.weights is not None:
                weights = paddle.static.data(
                    'weights', self.weights.shape, dtype=self.weights.dtype
                )
                out_0, out_1 = paddle.histogramdd(
                    x,
                    bins=self.bins,
                    weights=weights,
                    ranges=self.ranges,
                    density=self.density,
                )
            else:
                out_0, out_1 = paddle.histogramdd(
                    x, bins=self.bins, ranges=self.ranges, density=self.density
                )
            exe = paddle.static.Executor(self.place)
            if self.weights is not None:
                res = exe.run(
                    feed={'x': self.sample, 'weights': self.weights},
                    fetch_list=[out_0, out_1],
                )
            else:
                res = exe.run(
                    feed={'x': self.sample}, fetch_list=[out_0, out_1]
                )

            hist_out, edges_out = res[0], res[1:]
            np.testing.assert_allclose(
                hist_out,
                self.expect_hist,
            )
            for idx, edge_out in enumerate(edges_out):
                expect_edge = np.array(self.expect_edges[idx])
                np.testing.assert_allclose(
                    edge_out,
                    expect_edge,
                )

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        self.sample_dy = paddle.to_tensor(self.sample, dtype=self.sample.dtype)
        self.weights_dy = None
        if self.weights is not None:
            self.weights_dy = paddle.to_tensor(self.weights)
        if isinstance(self.bins, tuple):
            self.bins = tuple([paddle.to_tensor(bin) for bin in self.bins])
        hist, edges = paddle.histogramdd(
            self.sample_dy,
            bins=self.bins,
            weights=self.weights_dy,
            ranges=self.ranges,
            density=self.density,
        )

        np.testing.assert_allclose(
            hist.numpy(),
            self.expect_hist,
        )
        for idx, edge in enumerate(edges):
            edge = edge.numpy()
            expect_edge = np.array(self.expect_edges[idx])
            np.testing.assert_allclose(
                edge,
                expect_edge,
            )

        paddle.enable_static()

    def test_error(self):
        pass


class TestHistogramddAPICase1ForDensity(TestHistogramddAPI):
    def init_input(self):
        # self.sample = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        self.sample = np.random.randn(4, 2).astype(np.float64)
        self.bins = [2, 2]
        self.ranges = [0.0, 1.0, 0.0, 1.0]
        self.density = True

    def set_expect_output(self):
        self.expect_hist, self.expect_edges = ref_histogramdd(
            self.sample, self.bins, self.ranges, self.weights, self.density
        )


class TestHistogramddAPICase2ForMultiDimsAndDensity(TestHistogramddAPI):
    def init_input(self):
        # shape: [4,2,2]
        self.sample = np.random.randn(4, 2, 2).astype(np.float64)
        self.bins = [3, 4]
        self.density = True

    def set_expect_output(self):
        self.expect_hist, self.expect_edges = ref_histogramdd(
            self.sample, self.bins, self.ranges, self.weights, self.density
        )


class TestHistogramddAPICase3ForMultiDimsNotDensity(TestHistogramddAPI):
    def init_input(self):
        # shape: [4,2,2]
        self.sample = np.random.randn(4, 2, 2).astype(np.float64)
        self.bins = [3, 4]
        # self.density = True

    def set_expect_output(self):
        self.expect_hist, self.expect_edges = ref_histogramdd(
            self.sample, self.bins, self.ranges, self.weights, self.density
        )


class TestHistogramddAPICase4ForRangesAndDensity(TestHistogramddAPI):
    def init_input(self):
        # shape: [4,2,2]
        self.sample = np.random.randn(4, 2, 2).astype(np.float64)
        self.bins = [3, 4]
        # [leftmost_1, rightmost_1, leftmost_2, rightmost_2,..., leftmost_D, rightmost_D]
        self.ranges = [1.0, 10.0, 1.0, 100.0]
        self.density = True

    def set_expect_output(self):
        self.expect_hist, self.expect_edges = ref_histogramdd(
            self.sample, self.bins, self.ranges, self.weights, self.density
        )


class TestHistogramddAPICase5ForRangesNotDensity(TestHistogramddAPI):
    def init_input(self):
        # shape: [4,2,2]
        self.sample = np.random.randn(4, 2, 2).astype(np.float64)
        self.bins = [3, 4]
        # [leftmost_1, rightmost_1, leftmost_2, rightmost_2,..., leftmost_D, rightmost_D]
        self.ranges = [1.0, 10.0, 1.0, 100.0]
        # self.density = True

    def set_expect_output(self):
        self.expect_hist, self.expect_edges = ref_histogramdd(
            self.sample, self.bins, self.ranges, self.weights, self.density
        )


class TestHistogramddAPICase6NotRangesAndDensityAndWeights(TestHistogramddAPI):
    def init_input(self):
        # shape: [4,2,2]
        self.sample = np.random.randn(4, 2, 2).astype(np.float64)
        self.bins = [3, 4]
        # [leftmost_1, rightmost_1, leftmost_2, rightmost_2,..., leftmost_D, rightmost_D]
        # self.ranges = [1., 10., 1., 100.]
        self.density = True
        self.weights = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ],
            dtype=self.sample.dtype,
        )

    def set_expect_output(self):
        self.expect_hist, self.expect_edges = ref_histogramdd(
            self.sample, self.bins, self.ranges, self.weights, self.density
        )


class TestHistogramddAPICase7ForRangesAndDensityAndWeights(TestHistogramddAPI):
    def init_input(self):
        # shape: [4,2,2]
        self.sample = np.random.randn(4, 2, 2).astype(np.float64)
        self.bins = [3, 4]
        # [leftmost_1, rightmost_1, leftmost_2, rightmost_2,..., leftmost_D, rightmost_D]
        self.ranges = [1.0, 10.0, 1.0, 100.0]
        self.density = True
        self.weights = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ],
            dtype=self.sample.dtype,
        )

    def set_expect_output(self):
        self.expect_hist, self.expect_edges = ref_histogramdd(
            self.sample, self.bins, self.ranges, self.weights, self.density
        )


class TestHistogramddAPICase8MoreInnermostDim(TestHistogramddAPI):
    def init_input(self):
        # shape: [4,2,4]
        self.sample = np.random.randn(4, 2, 4).astype(np.float64)
        self.bins = [1, 2, 3, 4]
        # [leftmost_1, rightmost_1, leftmost_2, rightmost_2,..., leftmost_D, rightmost_D]
        self.density = False
        self.weights = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ],
            dtype=self.sample.dtype,
        )

    def set_expect_output(self):
        self.expect_hist, self.expect_edges = ref_histogramdd(
            self.sample, self.bins, self.ranges, self.weights, self.density
        )


class TestHistogramddAPICase8MoreInnermostDimAndDensity(TestHistogramddAPI):
    def init_input(self):
        # shape: [4,2,4]
        self.sample = np.random.randn(4, 2, 4).astype(np.float64)
        self.bins = [1, 2, 3, 4]
        # [leftmost_1, rightmost_1, leftmost_2, rightmost_2,..., leftmost_D, rightmost_D]
        self.density = True
        self.weights = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ],
            dtype=self.sample.dtype,
        )

    def set_expect_output(self):
        self.expect_hist, self.expect_edges = ref_histogramdd(
            self.sample, self.bins, self.ranges, self.weights, self.density
        )


class TestHistogramddAPICase9ForIntBin(TestHistogramddAPI):
    def init_input(self):
        # shape: [4,2,2]
        self.sample = np.random.randn(4, 2, 2).astype(np.float64)
        self.weights = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ],
            dtype=self.sample.dtype,
        )
        self.bins = 5
        self.density = True
        self.ranges = [1.0, 10.0, 1.0, 100.0]

    def set_expect_output(self):
        self.expect_hist, self.expect_edges = ref_histogramdd(
            self.sample, self.bins, self.ranges, self.weights, self.density
        )


class TestHistogramddAPICase10ForTensorBin(TestHistogramddAPI):
    def init_input(self):
        # shape: [4,2,2]
        self.sample = np.random.randn(4, 2, 2).astype(np.float64)
        self.weights = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ],
            dtype=self.sample.dtype,
        )
        self.bins = (
            np.array([1.0, 2.0, 10.0, 15.0, 20.0]),
            np.array([0.0, 20.0, 100.0]),
        )
        self.density = True

    def set_expect_output(self):
        self.expect_hist, self.expect_edges = ref_histogramdd(
            self.sample, self.bins, self.ranges, self.weights, self.density
        )


class TestHistogramddAPICase10ForFloat32(TestHistogramddAPI):
    def init_input(self):
        # self.sample = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        self.sample = np.random.randn(4, 2).astype(np.float32)
        self.bins = [2, 2]
        self.ranges = [0.0, 1.0, 0.0, 1.0]
        self.density = True

    def set_expect_output(self):
        self.expect_hist, self.expect_edges = ref_histogramdd(
            self.sample, self.bins, self.ranges, self.weights, self.density
        )


# histogramdd(sample, bins=10, ranges=None, density=False, weights=None, name=None):
class TestHistogramddAPI_check_sample_type_error(TestHistogramddAPI):
    def test_error(self):
        sample = paddle.to_tensor([[False, True], [True, False]])
        with self.assertRaises(TypeError):
            paddle.histogramdd(sample)


class TestHistogramddAPI_check_bins_element_error(TestHistogramddAPI):
    def test_error(self):
        sample = paddle.to_tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
                [[13.0, 14.0], [15.0, 16.0]],
            ]
        )
        bins = [3.4, 4.5]
        with self.assertRaises(ValueError):
            paddle.histogramdd(sample, bins=bins)


class TestHistogramddAPI_check_ranges_type_error(TestHistogramddAPI):
    def test_error(self):
        sample = paddle.to_tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
                [[13.0, 14.0], [15.0, 16.0]],
            ]
        )
        ranges = 10
        with self.assertRaises(TypeError):
            paddle.histogramdd(sample, ranges=ranges)


class TestHistogramddAPI_check_density_type_error(TestHistogramddAPI):
    def test_error(self):
        sample = paddle.to_tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
                [[13.0, 14.0], [15.0, 16.0]],
            ]
        )
        density = 10
        with self.assertRaises(TypeError):
            paddle.histogramdd(sample, density=density)


class TestHistogramddAPI_check_weights_type_error(TestHistogramddAPI):
    def test_error(self):
        sample = paddle.to_tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
                [[13.0, 14.0], [15.0, 16.0]],
            ]
        )
        weights = 10
        with self.assertRaises(AttributeError):
            paddle.histogramdd(sample, weights=weights)


class TestHistogramddAPI_sample_weights_shape_dismatch_error(
    TestHistogramddAPI
):
    def test_error(self):
        sample = paddle.to_tensor(
            [  # shape: [4,2]
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
                [[13.0, 14.0], [15.0, 16.0]],
            ]
        )
        weights = paddle.to_tensor(
            [2.0, 3.0, 4.0], dtype=self.sample.dtype
        )  # shape: [3,]
        with self.assertRaises(AssertionError):
            paddle.histogramdd(sample, weights=weights)


class TestHistogramddAPI_sample_weights_type_dismatch_error(TestHistogramddAPI):
    def test_error(self):
        sample = paddle.to_tensor(
            [  # float32
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
                [[13.0, 14.0], [15.0, 16.0]],
            ],
            dtype=paddle.float32,
        )
        weights = paddle.to_tensor(
            [2.0, 3.0, 4.0], dtype=paddle.float64
        )  # float64
        with self.assertRaises(AssertionError):
            paddle.histogramdd(sample, weights=weights)


class TestHistogramddAPI_check_bins_type_error(TestHistogramddAPI):
    def test_error(self):
        sample = paddle.to_tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
                [[13.0, 14.0], [15.0, 16.0]],
            ]
        )
        bins = 2.0
        with self.assertRaises(ValueError):
            paddle.histogramdd(sample, bins=bins)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
