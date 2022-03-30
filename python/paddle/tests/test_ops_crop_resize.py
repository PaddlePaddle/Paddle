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
from paddle.vision.ops import image_resize


def np_nearest_interp(image,
                      size,
                      align_corners=True,
                      data_format='NCHW'):
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


def np_image_resize(images, size, interp_method,
                    align_corners=True, align_mode=1,
                    data_format="NCHW"):
    if isinstance(size, int):
        size = (size, size)

    results = []
    if interp_method == "nearest":
        for image in images:
            results.append(np_nearest_interp(image,
                                             size=size,
                                             align_corners=align_corners,
                                             data_format=data_format))
    elif interp_method == "bilinear":
        for image in images:
            results.append(np_bilinear_interp(image,
                                              size=size,
                                              align_corners=align_corners,
                                              align_mode=align_mode,
                                              data_format=data_format))
    else:
        raise ValueError("unknown interp_method")

    return np.stack(results, axis=0)


class TestImageResizeNearestNCHW(unittest.TestCase):
    def setup(self):
        self.image_shape1 = [3, 8, 8]
        self.image_shape2 = [3, 2, 2]
        self.size = (4, 4)
        self.interp_method = "nearest"
        self.data_format = "NCHW"
        self.align_corners = False
        self.align_mode = 0

        self._is_np_built = False
        self.build_np_data()

    def build_np_data(self):
        if not self._is_np_built:
            self.image1 = np.random.randint(0, 256, self.image_shape1, dtype="uint8")
            self.image2 = np.random.randint(0, 256, self.image_shape2, dtype="uint8")
            self.np_result = np_image_resize(
                                [self.image1, self.image2],
                                size=self.size,
                                interp_method=self.interp_method,
                                align_corners=self.align_corners,
                                align_mode=self.align_mode,
                                data_format=self.data_format)
            self._is_np_built = True

    def test_output_dynamic(self):
        if not core.is_compiled_with_cuda():
            return

        paddle.disable_static()
        self.setup()

        images = paddle.tensor.create_array(dtype="uint8")
        images = paddle.tensor.array_write(paddle.to_tensor(self.image1), 
                                           paddle.to_tensor(0), images)
        images = paddle.tensor.array_write(paddle.to_tensor(self.image2),
                                           paddle.to_tensor(1), images)

        # NOTE: image_resize takes TensorArray as input, which cannot
        #       create by Python API in dynamic mode
        try:
            dy_result = image_resize(images, self.size,
                                     interp_method=self.interp_method,
                                     align_corners=self.align_corners,
                                     align_mode=self.align_mode,
                                     data_format=self.data_format)
        except:
            pass

    def test_output_static(self):
        if not core.is_compiled_with_cuda():
            return

        paddle.enable_static()
        self.setup()

        images = paddle.tensor.create_array(dtype="uint8")

        idx = fluid.layers.fill_constant(shape=[1], dtype="int64", value=0)
        image1 = fluid.layers.assign(self.image1.astype('int32'))
        image1 = fluid.layers.cast(image1, dtype='uint8')
        images = paddle.tensor.array_write(image1, idx, images)

        image2 = fluid.layers.assign(self.image2.astype('int32'))
        image2 = fluid.layers.cast(image2, dtype='uint8')
        images = paddle.tensor.array_write(image2, idx + 1, images)

        out = image_resize(images, self.size,
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
    def setup(self):
        self.image_shape1 = [32, 32, 3]
        self.image_shape2 = [16, 16, 3]
        self.size = 20
        self.interp_method = "nearest"
        self.data_format = "NHWC"
        self.align_corners = True
        self.align_mode = 1

        self._is_np_built = False
        self.build_np_data()

    def test_output_dynamic(self):
        pass


class TestImageResizeNearestNCHWAlignCorner(TestImageResizeNearestNHWC):
    def setup(self):
        self.image_shape1 = [3, 32, 32]
        self.image_shape2 = [3, 16, 16]
        self.size = 30
        self.interp_method = "nearest"
        self.data_format = "NCHW"
        self.align_corners = True
        self.align_mode = 1

        self._is_np_built = False
        self.build_np_data()


class TestImageResizeNearestNHWCAlignCorner(TestImageResizeNearestNHWC):
    def setup(self):
        self.image_shape1 = [32, 32, 3]
        self.image_shape2 = [16, 16, 3]
        self.size = (20, 30)
        self.interp_method = "nearest"
        self.data_format = "NHWC"
        self.align_corners = True
        self.align_mode = 1

        self._is_np_built = False
        self.build_np_data()


class TestImageResizeBilinearNCHW(TestImageResizeNearestNHWC):
    def setup(self):
        self.image_shape1 = [3, 32, 32]
        self.image_shape2 = [3, 16, 16]
        self.size = (20, 30)
        self.interp_method = "bilinear"
        self.data_format = "NCHW"
        self.align_corners = False
        self.align_mode = 1

        self._is_np_built = False
        self.build_np_data()


class TestImageResizeBilinearNHWC(TestImageResizeNearestNHWC):
    def setup(self):
        self.image_shape1 = [32, 32, 3]
        self.image_shape2 = [16, 16, 3]
        self.size = (20, 30)
        self.interp_method = "bilinear"
        self.data_format = "NHWC"
        self.align_corners = False
        self.align_mode = 1

        self._is_np_built = False
        self.build_np_data()


class TestImageResizeBilinearNCHWAlignMode0(TestImageResizeNearestNHWC):
    def setup(self):
        self.image_shape1 = [3, 32, 32]
        self.image_shape2 = [3, 16, 16]
        self.size = (20, 30)
        self.interp_method = "bilinear"
        self.data_format = "NCHW"
        self.align_corners = False
        self.align_mode = 0

        self._is_np_built = False
        self.build_np_data()


class TestImageResizeBilinearNHWCAlignMode0(TestImageResizeNearestNHWC):
    def setup(self):
        self.image_shape1 = [32, 32, 3]
        self.image_shape2 = [16, 16, 3]
        self.size = (20, 30)
        self.interp_method = "bilinear"
        self.data_format = "NHWC"
        self.align_corners = False
        self.align_mode = 0

        self._is_np_built = False
        self.build_np_data()


class TestImageResizeBilinearNCHWAlignCorner(TestImageResizeNearestNHWC):
    def setup(self):
        self.image_shape1 = [3, 32, 32]
        self.image_shape2 = [3, 16, 16]
        self.size = (20, 30)
        self.interp_method = "bilinear"
        self.data_format = "NCHW"
        self.align_corners = True
        self.align_mode = 1

        self._is_np_built = False
        self.build_np_data()


class TestImageResizeBilinearNHWCAlignCorner(TestImageResizeNearestNHWC):
    def setup(self):
        self.image_shape1 = [32, 32, 3]
        self.image_shape2 = [16, 16, 3]
        self.size = (20, 30)
        self.interp_method = "bilinear"
        self.data_format = "NHWC"
        self.align_corners = True
        self.align_mode = 1

        self._is_np_built = False
        self.build_np_data()


if __name__ == '__main__':
    unittest.main()
