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

import math
import os
import unittest

import numpy as np
from op_test import OpTest

import paddle


def calc_psroi_pool(
    x,
    rois,
    rois_num_per_img,
    output_channels,
    spatial_scale,
    pooled_height,
    pooled_width,
):
    """
    Psroi_pool implemented by Numpy.
    x: 4-D as (N, C, H, W),
    rois: 2-D as [[x1, y1, x2, y2], ...],
    rois_num_per_img: 1-D as [nums_of_batch_0, nums_of_batch_1,  ...]
    """
    output_shape = (len(rois), output_channels, pooled_height, pooled_width)
    out_data = np.zeros(output_shape)
    batch_id = 0
    rois_num_id = 0
    rois_num_left = rois_num_per_img[rois_num_id]
    for i in range(len(rois)):
        roi = rois[i]
        roi_batch_id = batch_id
        rois_num_left -= 1
        if rois_num_left == 0:
            rois_num_id += 1
            if rois_num_id < len(rois_num_per_img):
                rois_num_left = rois_num_per_img[rois_num_id]
            batch_id += 1
        roi_start_w = round(roi[0]) * spatial_scale
        roi_start_h = round(roi[1]) * spatial_scale
        roi_end_w = (round(roi[2]) + 1.0) * spatial_scale
        roi_end_h = (round(roi[3]) + 1.0) * spatial_scale

        roi_height = max(roi_end_h - roi_start_h, 0.1)
        roi_width = max(roi_end_w - roi_start_w, 0.1)

        bin_size_h = roi_height / float(pooled_height)
        bin_size_w = roi_width / float(pooled_width)

        x_i = x[roi_batch_id]

        for c in range(output_channels):
            for ph in range(pooled_height):
                for pw in range(pooled_width):
                    hstart = int(
                        math.floor(float(ph) * bin_size_h + roi_start_h)
                    )
                    wstart = int(
                        math.floor(float(pw) * bin_size_w + roi_start_w)
                    )
                    hend = int(
                        math.ceil(float(ph + 1) * bin_size_h + roi_start_h)
                    )
                    wend = int(
                        math.ceil(float(pw + 1) * bin_size_w + roi_start_w)
                    )
                    hstart = min(max(hstart, 0), x.shape[2])
                    hend = min(max(hend, 0), x.shape[2])
                    wstart = min(max(wstart, 0), x.shape[3])
                    wend = min(max(wend, 0), x.shape[3])

                    c_in = (c * pooled_height + ph) * pooled_width + pw
                    is_empty = (hend <= hstart) or (wend <= wstart)
                    out_sum = 0.0
                    for ih in range(hstart, hend):
                        for iw in range(wstart, wend):
                            out_sum += x_i[c_in, ih, iw]
                    bin_area = (hend - hstart) * (wend - wstart)
                    out_data[i, c, ph, pw] = (
                        0.0 if is_empty else (out_sum / float(bin_area))
                    )
    return out_data


class TestPSROIPoolOp(OpTest):
    def set_data(self):
        paddle.enable_static()
        self.init_test_case()
        self.make_rois()
        self.outs = calc_psroi_pool(
            self.x,
            self.boxes,
            self.boxes_num,
            self.output_channels,
            self.spatial_scale,
            self.pooled_height,
            self.pooled_width,
        ).astype('float64')
        self.inputs = {
            'X': self.x,
            'ROIs': (self.rois_with_batch_id[:, 1:5], self.rois_lod),
            'RoisNum': self.boxes_num,
        }
        self.attrs = {
            'output_channels': self.output_channels,
            'spatial_scale': self.spatial_scale,
            'pooled_height': self.pooled_height,
            'pooled_width': self.pooled_width,
        }
        self.outputs = {'Out': self.outs}

    def init_test_case(self):
        self.batch_size = 3
        self.channels = 3 * 2 * 2
        self.height = 6
        self.width = 4

        self.x_dim = [self.batch_size, self.channels, self.height, self.width]

        self.spatial_scale = 1.0 / 4.0
        self.output_channels = 3
        self.pooled_height = 2
        self.pooled_width = 2

        self.x = np.random.random(self.x_dim).astype('float64')

    def make_rois(self):
        rois = []
        self.rois_lod = [[]]
        for bno in range(self.batch_size):
            self.rois_lod[0].append(bno + 1)
            for i in range(bno + 1):
                x1 = np.random.random_integers(
                    0, self.width // self.spatial_scale - self.pooled_width
                )
                y1 = np.random.random_integers(
                    0, self.height // self.spatial_scale - self.pooled_height
                )

                x2 = np.random.random_integers(
                    x1 + self.pooled_width, self.width // self.spatial_scale
                )
                y2 = np.random.random_integers(
                    y1 + self.pooled_height, self.height // self.spatial_scale
                )
                roi = [bno, x1, y1, x2, y2]
                rois.append(roi)
        self.rois_num = len(rois)
        self.rois_with_batch_id = np.array(rois).astype('float64')
        self.boxes = self.rois_with_batch_id[:, 1:]
        self.boxes_num = np.array(
            [bno + 1 for bno in range(self.batch_size)]
        ).astype('int32')

    def setUp(self):
        self.op_type = 'psroi_pool'
        self.python_api = lambda x, boxes, boxes_num, pooled_height, pooled_width, output_channels, spatial_scale: paddle.vision.ops.psroi_pool(
            x, boxes, boxes_num, (pooled_height, pooled_width), spatial_scale
        )
        self.set_data()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestPSROIPoolDynamicFunctionAPI(unittest.TestCase):
    def setUp(self):
        self.x = np.random.random([2, 490, 28, 28]).astype(np.float32)
        self.boxes = np.array(
            [[1, 5, 8, 10], [4, 2, 6, 7], [12, 12, 19, 21]]
        ).astype(np.float32)
        self.boxes_num = np.array([1, 2]).astype(np.int32)

    def test_output_size(self):
        def test_output_size_is_int():
            output_size = 7
            out = paddle.vision.ops.psroi_pool(
                paddle.to_tensor(self.x),
                paddle.to_tensor(self.boxes),
                paddle.to_tensor(self.boxes_num),
                output_size,
            ).numpy()
            expect_out = calc_psroi_pool(
                self.x, self.boxes, self.boxes_num, 10, 1.0, 7, 7
            )
            np.testing.assert_allclose(out, expect_out, rtol=1e-05)

        def test_output_size_is_tuple():
            output_size = (7, 7)
            out = paddle.vision.ops.psroi_pool(
                paddle.to_tensor(self.x),
                paddle.to_tensor(self.boxes),
                paddle.to_tensor(self.boxes_num),
                output_size,
            ).numpy()
            expect_out = calc_psroi_pool(
                self.x, self.boxes, self.boxes_num, 10, 1.0, 7, 7
            )
            np.testing.assert_allclose(out, expect_out, rtol=1e-05)

        def test_dytype_is_float64():
            output_size = (7, 7)
            out = paddle.vision.ops.psroi_pool(
                paddle.to_tensor(self.x, 'float64'),
                paddle.to_tensor(self.boxes, 'float64'),
                paddle.to_tensor(self.boxes_num, 'int32'),
                output_size,
            ).numpy()
            expect_out = calc_psroi_pool(
                self.x, self.boxes, self.boxes_num, 10, 1.0, 7, 7
            )
            np.testing.assert_allclose(out, expect_out, rtol=1e-05)

        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.base.core.is_compiled_with_cuda()
        ):
            places.append('cpu')
        if paddle.base.core.is_compiled_with_cuda():
            places.append('gpu')
        for place in places:
            paddle.set_device(place)
            test_output_size_is_int()
            test_output_size_is_tuple()
            test_dytype_is_float64()


class TestPSROIPoolDynamicClassAPI(unittest.TestCase):
    def setUp(self):
        self.x = np.random.random([2, 128, 32, 32]).astype(np.float32)
        self.boxes = np.array(
            [[3, 5, 6, 13], [7, 4, 22, 18], [4, 5, 7, 10], [5, 3, 25, 21]]
        ).astype(np.float32)
        self.boxes_num = np.array([2, 2]).astype(np.int32)

    def test_output_size(self):
        def test_output_size_is_int():
            psroi_module = paddle.vision.ops.PSRoIPool(8, 1.1)
            out = psroi_module(
                paddle.to_tensor(self.x),
                paddle.to_tensor(self.boxes),
                paddle.to_tensor(self.boxes_num),
            ).numpy()
            expect_out = calc_psroi_pool(
                self.x, self.boxes, self.boxes_num, 2, 1.1, 8, 8
            )
            np.testing.assert_allclose(out, expect_out, rtol=1e-05)

        def test_output_size_is_tuple():
            psroi_pool_module = paddle.vision.ops.PSRoIPool(8, 1.1)
            out = psroi_pool_module(
                paddle.to_tensor(self.x),
                paddle.to_tensor(self.boxes),
                paddle.to_tensor(self.boxes_num),
            ).numpy()
            expect_out = calc_psroi_pool(
                self.x, self.boxes, self.boxes_num, 2, 1.1, 8, 8
            )
            np.testing.assert_allclose(out, expect_out, rtol=1e-05)

        def test_dytype_is_float64():
            psroi_pool_module = paddle.vision.ops.PSRoIPool(8, 1.1)
            out = psroi_pool_module(
                paddle.to_tensor(self.x, 'float64'),
                paddle.to_tensor(self.boxes, 'float64'),
                paddle.to_tensor(self.boxes_num, 'int32'),
            ).numpy()
            expect_out = calc_psroi_pool(
                self.x, self.boxes, self.boxes_num, 2, 1.1, 8, 8
            )
            np.testing.assert_allclose(out, expect_out, rtol=1e-05)

        paddle.disable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.base.core.is_compiled_with_cuda()
        ):
            places.append('cpu')
        if paddle.base.core.is_compiled_with_cuda():
            places.append('gpu')
        for place in places:
            paddle.set_device(place)
            test_output_size_is_int()
            test_output_size_is_tuple()
            test_dytype_is_float64()


class TestPSROIPoolBoxesNumError(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.x = paddle.uniform([2, 490, 28, 28], dtype='float32')
        self.boxes = paddle.to_tensor(
            [[1, 5, 8, 10], [4, 2, 6, 7], [12, 12, 19, 21]], 'float32'
        )

    def test_errors(self):
        def test_boxes_num_nums_error():
            boxes_num = paddle.to_tensor([1, 5], 'int32')
            out = paddle.vision.ops.psroi_pool(
                self.x, self.boxes, boxes_num, output_size=7
            )

        self.assertRaises(ValueError, test_boxes_num_nums_error)

        def test_boxes_num_length_error():
            boxes_num = paddle.to_tensor([1, 1, 1], 'int32')
            out = paddle.vision.ops.psroi_pool(
                self.x, self.boxes, boxes_num, output_size=7
            )

        self.assertRaises(ValueError, test_boxes_num_length_error)


class TestPSROIPoolChannelError(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.x = paddle.uniform([2, 490, 28, 28], dtype='float32')
        self.boxes = paddle.to_tensor(
            [[1, 5, 8, 10], [4, 2, 6, 7], [12, 12, 19, 21]], 'float32'
        )
        self.output_size = 4

    def test_errors(self):
        def test_channel_error():
            boxes_num = paddle.to_tensor([2, 1], 'int32')
            out = paddle.vision.ops.psroi_pool(
                self.x, self.boxes, boxes_num, self.output_size
            )

        self.assertRaises(ValueError, test_channel_error)


class TestPSROIPoolZeroDivError(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.x = paddle.uniform([2, 490, 28, 28], dtype='float32')
        self.boxes = paddle.to_tensor(
            [[1, 5, 8, 10], [4, 2, 6, 7], [12, 12, 19, 21]], dtype='float32'
        )
        self.boxes_num = paddle.to_tensor([1, 2], dtype='int32')

    def test_errors(self):
        def test_zero_div_error():
            paddle.vision.ops.psroi_pool(self.x, self.boxes, self.boxes_num, 0)

        self.assertRaises(ValueError, test_zero_div_error)


class TestPSROIPoolStaticAPI(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.x_placeholder = paddle.static.data(
            name='x', shape=[2, 490, 28, 28]
        )
        self.x = np.random.random([2, 490, 28, 28]).astype(np.float32)
        self.boxes_placeholder = paddle.static.data(name='boxes', shape=[3, 4])
        self.boxes = np.array(
            [[1, 5, 8, 10], [4, 2, 6, 7], [12, 12, 19, 21]]
        ).astype(np.float32)
        self.boxes_num = np.array([1, 2]).astype(np.int32)

    def test_function_in_static(self):
        output_size = 7
        out = paddle.vision.ops.psroi_pool(
            self.x_placeholder,
            self.boxes_placeholder,
            paddle.to_tensor(self.boxes_num),
            output_size,
        )
        expect_out = calc_psroi_pool(
            self.x, self.boxes, self.boxes_num, 10, 1.0, 7, 7
        )
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.base.core.is_compiled_with_cuda()
        ):
            places.append(paddle.CPUPlace())
        if paddle.base.core.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for place in places:
            exe = paddle.static.Executor(place)
            boxes_lod_data = paddle.base.create_lod_tensor(
                self.boxes, [[1, 2]], place
            )
            (out_res,) = exe.run(
                paddle.static.default_main_program(),
                feed={'x': self.x, 'boxes': boxes_lod_data},
                fetch_list=[out],
            )
            np.testing.assert_allclose(out_res, expect_out, rtol=1e-05)


class TestPSROIPoolStaticAPI_NOLOD(unittest.TestCase):
    def setUp(self):
        self.x = np.random.random([2, 490, 28, 28]).astype(np.float32)
        self.boxes = np.array(
            [[1, 5, 8, 10], [4, 2, 6, 7], [12, 12, 19, 21]]
        ).astype(np.float32)
        self.boxes_num = np.array([1, 2]).astype(np.int32)

    def test_function_in_pir(self):
        with paddle.pir_utils.IrGuard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                output_size = 7
                x_placeholder = paddle.static.data(
                    name='x', shape=[2, 490, 28, 28]
                )
                boxes_placeholder = paddle.static.data(
                    name='boxes_nolod', shape=[3, 4]
                )
                boxes_num = paddle.to_tensor(self.boxes_num, 'int32')
                out = paddle.vision.ops.psroi_pool(
                    x_placeholder,
                    boxes_placeholder,
                    boxes_num,
                    output_size,
                )
                expect_out = calc_psroi_pool(
                    self.x, self.boxes, self.boxes_num, 10, 1.0, 7, 7
                )
                places = []
                if (
                    os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
                    in ['1', 'true', 'on']
                    or not paddle.base.core.is_compiled_with_cuda()
                ):
                    places.append(paddle.CPUPlace())
                if paddle.base.core.is_compiled_with_cuda():
                    places.append(paddle.CUDAPlace(0))
                for place in places:
                    exe = paddle.static.Executor(place)
                    (out_res,) = exe.run(
                        paddle.static.default_main_program(),
                        feed={'x': self.x, 'boxes_nolod': self.boxes},
                        fetch_list=[out],
                    )
                    np.testing.assert_allclose(out_res, expect_out, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
