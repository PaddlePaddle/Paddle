# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import cv2
import unittest
import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.vision.ops import image_resize, random_crop_and_resize


def np_nearest_interp(image, size, align_corners=True, data_format='NCHW'):
    """nearest neighbor interpolation implement in shape [N, C, H, W]"""
    if isinstance(size, int):
        size = (size, size)

    if data_format == "NHWC":
        image = np.transpose(image, (2, 0, 1))  # HWC => CHW

    channel, in_h, in_w = image.shape
    out_h, out_w = size

    ratio_h = ratio_w = 0.0
    if (out_h > 1):
        if (align_corners):
            ratio_h = (in_h - 1.0) / (out_h - 1.0)
        else:
            ratio_h = 1.0 * in_h / out_h
    if (out_w > 1):
        if (align_corners):
            ratio_w = (in_w - 1.0) / (out_w - 1.0)
        else:
            ratio_w = 1.0 * in_w / out_w

    out = np.zeros((channel, out_h, out_w))

    if align_corners:
        for i in range(out_h):
            in_i = int(ratio_h * i + 0.5)
            for j in range(out_w):
                in_j = int(ratio_w * j + 0.5)
                out[:, i, j] = image[:, in_i, in_j]
    else:
        for i in range(out_h):
            in_i = int(ratio_h * i)
            for j in range(out_w):
                in_j = int(ratio_w * j)
                out[:, i, j] = image[:, in_i, in_j]

    if data_format == "NHWC":
        out = np.transpose(out, (1, 2, 0))  # CHW => HWC

    return out.astype(image.dtype)


def np_bilinear_interp(image,
                       size,
                       align_corners=True,
                       align_mode=0,
                       data_format='NCHW'):
    """bilinear interpolation implement in shape [N, C, H, W]"""
    if isinstance(size, int):
        size = (size, size)

    if data_format == "NHWC":
        image = np.transpose(image, (2, 0, 1))  # HWC => CHW

    channel, in_h, in_w = image.shape
    out_h, out_w = size

    ratio_h = ratio_w = 0.0
    if out_h > 1:
        if (align_corners):
            ratio_h = (in_h - 1.0) / (out_h - 1.0)
        else:
            ratio_h = 1.0 * in_h / out_h
    if out_w > 1:
        if (align_corners):
            ratio_w = (in_w - 1.0) / (out_w - 1.0)
        else:
            ratio_w = 1.0 * in_w / out_w

    out = np.zeros((channel, out_h, out_w))

    for i in range(out_h):
        if (align_mode == 0 and not align_corners):
            h = int(ratio_h * (i + 0.5) - 0.5)
        else:
            h = int(ratio_h * i)

        h = max(0, h)
        hid = 1 if h < in_h - 1 else 0
        if (align_mode == 0 and not align_corners):
            idx_src_h = max(ratio_h * (i + 0.5) - 0.5, 0)
            h1lambda = idx_src_h - h
        else:
            h1lambda = ratio_h * i - h
        h2lambda = 1.0 - h1lambda
        for j in range(out_w):
            if (align_mode == 0 and not align_corners):
                w = int(ratio_w * (j + 0.5) - 0.5)
            else:
                w = int(ratio_w * j)
            w = max(0, w)
            wid = 1 if w < in_w - 1 else 0
            if (align_mode == 0 and not align_corners):
                idx_src_w = max(ratio_w * (j + 0.5) - 0.5, 0)
                w1lambda = idx_src_w - w
            else:
                w1lambda = ratio_w * j - w
            w2lambda = 1.0 - w1lambda

            out[:, i, j] = h2lambda*(w2lambda * image[:, h, w] +
                                        w1lambda * image[:, h, w+wid]) + \
                h1lambda*(w2lambda * image[:, h+hid, w] +
                          w1lambda * image[:, h+hid, w+wid])

    if data_format == "NHWC":
        out = np.transpose(out, (1, 2, 0))  # CHW => HWC

    return out.astype(image.dtype)


def np_image_resize(images,
                    size,
                    interp_method,
                    align_corners=True,
                    align_mode=1,
                    data_format="NCHW"):
    if isinstance(size, int):
        size = (size, size)

    results = []
    if interp_method == "nearest":
        for image in images:
            results.append(
                np_nearest_interp(
                    image,
                    size=size,
                    align_corners=align_corners,
                    data_format=data_format))
    elif interp_method == "bilinear":
        for image in images:
            results.append(
                np_bilinear_interp(
                    image,
                    size=size,
                    align_corners=align_corners,
                    align_mode=align_mode,
                    data_format=data_format))
    else:
        raise ValueError("unknown interp_method")

    return np.stack(results, axis=0)


class TestImageResizeNearestNCHW(unittest.TestCase):
    def setUp(self):
        self.image_shape1 = [3, 32, 32]
        self.image_shape2 = [3, 16, 16]
        self.size = (20, 30)
        self.interp_method = "nearest"
        self.data_format = "NCHW"
        self.align_corners = False
        self.align_mode = 1

        self.build_np_data()

    def build_np_data(self):
        self.image1 = np.random.randint(
            0, 256, self.image_shape1, dtype="uint8")
        self.image2 = np.random.randint(
            0, 256, self.image_shape2, dtype="uint8")
        self.np_result = np_image_resize(
            [self.image1, self.image2],
            size=self.size,
            interp_method=self.interp_method,
            align_corners=self.align_corners,
            align_mode=self.align_mode,
            data_format=self.data_format)

    def test_output_dynamic(self):
        # NOTE: only support cuda kernel currently
        if not core.is_compiled_with_cuda():
            return

        paddle.disable_static()

        images = paddle.tensor.create_array(dtype="uint8")
        images = paddle.tensor.array_write(
            paddle.to_tensor(self.image1), paddle.to_tensor(0), images)
        images = paddle.tensor.array_write(
            paddle.to_tensor(self.image2), paddle.to_tensor(1), images)

        result = image_resize(
            images,
            self.size,
            interp_method=self.interp_method,
            align_corners=self.align_corners,
            align_mode=self.align_mode,
            data_format=self.data_format)
        assert np.allclose(result.numpy(), self.np_result, rtol=1)

    def test_output_static(self):
        # NOTE: only support cuda kernel currently
        if not core.is_compiled_with_cuda():
            return

        paddle.enable_static()

        image1 = fluid.layers.assign(self.image1.astype('int32'))
        image1 = fluid.layers.cast(image1, dtype='uint8')

        image2 = fluid.layers.assign(self.image2.astype('int32'))
        image2 = fluid.layers.cast(image2, dtype='uint8')

        out = image_resize(
            [image1, image2],
            self.size,
            interp_method=self.interp_method,
            align_corners=self.align_corners,
            align_mode=self.align_mode,
            data_format=self.data_format)

        exe = paddle.static.Executor(paddle.CUDAPlace(0))
        result, = exe.run(paddle.static.default_main_program(),
                          fetch_list=[out])
        assert np.allclose(result, self.np_result, rtol=1)

        paddle.disable_static()


class TestImageResizeNearestNHWC(TestImageResizeNearestNCHW):
    def setUp(self):
        self.image_shape1 = [32, 32, 3]
        self.image_shape2 = [16, 16, 3]
        self.size = 20
        self.interp_method = "nearest"
        self.data_format = "NHWC"
        self.align_corners = True
        self.align_mode = 1

        self.build_np_data()


class TestImageResizeNearestNCHWAlignCorner(TestImageResizeNearestNHWC):
    def setUp(self):
        self.image_shape1 = [3, 32, 32]
        self.image_shape2 = [3, 16, 16]
        self.size = 30
        self.interp_method = "nearest"
        self.data_format = "NCHW"
        self.align_corners = True
        self.align_mode = 1

        self.build_np_data()


class TestImageResizeNearestNHWCAlignCorner(TestImageResizeNearestNHWC):
    def setUp(self):
        self.image_shape1 = [32, 32, 3]
        self.image_shape2 = [16, 16, 3]
        self.size = (20, 30)
        self.interp_method = "nearest"
        self.data_format = "NHWC"
        self.align_corners = True
        self.align_mode = 1

        self.build_np_data()


class TestImageResizeBilinearNCHW(TestImageResizeNearestNHWC):
    def setUp(self):
        self.image_shape1 = [3, 32, 32]
        self.image_shape2 = [3, 16, 16]
        self.size = (20, 30)
        self.interp_method = "bilinear"
        self.data_format = "NCHW"
        self.align_corners = False
        self.align_mode = 1

        self.build_np_data()


class TestImageResizeBilinearNHWC(TestImageResizeNearestNHWC):
    def setUp(self):
        self.image_shape1 = [32, 32, 3]
        self.image_shape2 = [16, 16, 3]
        self.size = (20, 30)
        self.interp_method = "bilinear"
        self.data_format = "NHWC"
        self.align_corners = False
        self.align_mode = 1

        self.build_np_data()


class TestImageResizeBilinearNCHWAlignMode0(TestImageResizeNearestNHWC):
    def setUp(self):
        self.image_shape1 = [3, 32, 32]
        self.image_shape2 = [3, 16, 16]
        self.size = (20, 30)
        self.interp_method = "bilinear"
        self.data_format = "NCHW"
        self.align_corners = False
        self.align_mode = 0

        self.build_np_data()


class TestImageResizeBilinearNHWCAlignMode0(TestImageResizeNearestNHWC):
    def setUp(self):
        self.image_shape1 = [32, 32, 3]
        self.image_shape2 = [16, 16, 3]
        self.size = (20, 30)
        self.interp_method = "bilinear"
        self.data_format = "NHWC"
        self.align_corners = False
        self.align_mode = 0

        self.build_np_data()


class TestImageResizeBilinearNCHWAlignCorner(TestImageResizeNearestNHWC):
    def setUp(self):
        self.image_shape1 = [3, 32, 32]
        self.image_shape2 = [3, 16, 16]
        self.size = (20, 30)
        self.interp_method = "bilinear"
        self.data_format = "NCHW"
        self.align_corners = True
        self.align_mode = 1

        self.build_np_data()


class TestImageResizeBilinearNHWCAlignCorner(TestImageResizeNearestNHWC):
    def setUp(self):
        self.image_shape1 = [32, 32, 3]
        self.image_shape2 = [16, 16, 3]
        self.size = (20, 30)
        self.interp_method = "bilinear"
        self.data_format = "NHWC"
        self.align_corners = True
        self.align_mode = 1

        self.build_np_data()


class TestImageCropResizeNearestNCHW(unittest.TestCase):
    def setUp(self):
        self.image_shape1 = [3, 16, 16]
        self.image_shape2 = [3, 32, 32]
        self.size = (20, 30)
        self.interp_method = "nearest"
        self.data_format = "NCHW"
        self.align_corners = False
        self.align_mode = 1

        self.out_shape = (2, 3, 20, 30)

        self.build_np_data()

    def build_np_data(self):
        self.image1 = np.random.randint(
            0, 256, self.image_shape1, dtype="uint8")
        self.image2 = np.random.randint(
            0, 256, self.image_shape2, dtype="uint8")

    def test_output_dynamic(self):
        # NOTE: only support cuda kernel currently
        if not core.is_compiled_with_cuda():
            return

        paddle.disable_static()

        images = paddle.tensor.create_array(dtype="uint8")
        images = paddle.tensor.array_write(
            paddle.to_tensor(self.image1), paddle.to_tensor(0), images)
        images = paddle.tensor.array_write(
            paddle.to_tensor(self.image2), paddle.to_tensor(1), images)

        result = random_crop_and_resize(
            images,
            self.size,
            interp_method=self.interp_method,
            align_corners=self.align_corners,
            align_mode=self.align_mode,
            data_format=self.data_format)
        result = result.numpy()
        assert result.shape == self.out_shape
        assert result.dtype == np.uint8

    def test_output_static(self):
        # NOTE: only support cuda kernel currently
        if not core.is_compiled_with_cuda():
            return

        paddle.enable_static()

        image1 = fluid.layers.assign(self.image1.astype('int32'))
        image1 = fluid.layers.cast(image1, dtype='uint8')

        image2 = fluid.layers.assign(self.image2.astype('int32'))
        image2 = fluid.layers.cast(image2, dtype='uint8')

        out = random_crop_and_resize(
            [image1, image2],
            self.size,
            interp_method=self.interp_method,
            align_corners=self.align_corners,
            align_mode=self.align_mode,
            data_format=self.data_format)

        exe = paddle.static.Executor(paddle.CUDAPlace(0))
        result, = exe.run(paddle.static.default_main_program(),
                          fetch_list=[out])
        assert result.shape == self.out_shape
        assert result.dtype == np.uint8

        paddle.disable_static()


class TestImageCropResizeNearestNHWC(TestImageCropResizeNearestNCHW):
    def setUp(self):
        self.image_shape1 = [16, 16, 3]
        self.image_shape2 = [32, 32, 3]
        self.size = 20
        self.interp_method = "nearest"
        self.data_format = "NHWC"
        self.align_corners = False
        self.align_mode = 1

        self.out_shape = (2, 20, 20, 3)

        self.build_np_data()


class TestImageCropResizeNearestNCHWAlignCorner(TestImageCropResizeNearestNCHW):
    def setUp(self):
        self.image_shape1 = [3, 16, 16]
        self.image_shape2 = [3, 32, 32]
        self.size = 20
        self.interp_method = "nearest"
        self.data_format = "NCHW"
        self.align_corners = True
        self.align_mode = 1

        self.out_shape = (2, 3, 20, 20)

        self.build_np_data()


class TestImageCropResizeNearestNHWCAlignCorner(TestImageCropResizeNearestNCHW):
    def setUp(self):
        self.image_shape1 = [16, 16, 3]
        self.image_shape2 = [32, 32, 3]
        self.size = (20, 30)
        self.interp_method = "nearest"
        self.data_format = "NHWC"
        self.align_corners = True
        self.align_mode = 1

        self.out_shape = (2, 20, 30, 3)

        self.build_np_data()


class TestImageCropResizeBilinearNCHW(TestImageCropResizeNearestNCHW):
    def setUp(self):
        self.image_shape1 = [3, 16, 16]
        self.image_shape2 = [3, 32, 32]
        self.size = (20, 30)
        self.interp_method = "bilinear"
        self.data_format = "NCHW"
        self.align_corners = False
        self.align_mode = 1

        self.out_shape = (2, 3, 20, 30)

        self.build_np_data()


class TestImageCropResizeBilinearNCHWAlignMode0(TestImageCropResizeNearestNCHW):
    def setUp(self):
        self.image_shape1 = [3, 16, 16]
        self.image_shape2 = [3, 32, 32]
        self.size = (20, 30)
        self.interp_method = "bilinear"
        self.data_format = "NCHW"
        self.align_corners = False
        self.align_mode = 0

        self.out_shape = (2, 3, 20, 30)

        self.build_np_data()


class TestImageCropResizeNearestNHWCAlignMode0(TestImageCropResizeNearestNCHW):
    def setUp(self):
        self.image_shape1 = [16, 16, 3]
        self.image_shape2 = [32, 32, 3]
        self.size = (20, 30)
        self.interp_method = "bilinear"
        self.data_format = "NHWC"
        self.align_corners = False
        self.align_mode = 0

        self.out_shape = (2, 20, 30, 3)

        self.build_np_data()


class TestImageCropResizeBilinearNCHWAlignCorner(
        TestImageCropResizeNearestNCHW):
    def setUp(self):
        self.image_shape1 = [3, 16, 16]
        self.image_shape2 = [3, 32, 32]
        self.size = (20, 30)
        self.interp_method = "bilinear"
        self.data_format = "NCHW"
        self.align_corners = True
        self.align_mode = 1

        self.out_shape = (2, 3, 20, 30)

        self.build_np_data()


if __name__ == '__main__':
    unittest.main()
