# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import tempfile
import unittest

import cv2
import numpy as np
from op_test import paddle_static_guard

import paddle
from paddle.vision.ops import decode_jpeg, read_file


class TestReadFileWithDynamic(unittest.TestCase):
    def setUp(self):
        fake_img = (np.random.random((400, 300, 3)) * 255).astype('uint8')
        self.temp_dir = tempfile.TemporaryDirectory()
        self.img_path = os.path.join(self.temp_dir.name, 'fake.jpg')
        cv2.imwrite(self.img_path, fake_img)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_read_file_decode_jpeg_dynamic(self):
        if not paddle.is_compiled_with_cuda():
            return
        img_bytes = read_file(self.img_path)
        img = decode_jpeg(img_bytes, mode='gray')
        img = decode_jpeg(img_bytes, mode='rgb')
        img = decode_jpeg(img_bytes)
        img_cv2 = cv2.imread(self.img_path)
        np.testing.assert_equal(img.shape, img_cv2.transpose(2, 0, 1).shape)


class TestReadFileWithStatic(unittest.TestCase):
    def setUp(self):
        fake_img = (np.random.random((400, 300, 3)) * 255).astype('uint8')
        self.temp_dir = tempfile.TemporaryDirectory()
        self.img_path = os.path.join(self.temp_dir.name, 'fake.jpg')
        cv2.imwrite(self.img_path, fake_img)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_read_file_decode_jpeg_static(self):
        paddle.enable_static()
        if not paddle.is_compiled_with_cuda():
            return
        place = paddle.CUDAPlace(0)
        with paddle_static_guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                img_bytes = read_file(self.img_path)
                img = decode_jpeg(img_bytes, mode='gray')
                img = decode_jpeg(img_bytes, mode='rgb')
                img = decode_jpeg(img_bytes)
                img_cv2 = cv2.imread(self.img_path)
                exe = paddle.static.Executor(place)
                out = exe.run(
                    paddle.static.default_main_program(), fetch_list=[img]
                )

                np.testing.assert_equal(
                    out[0].shape, img_cv2.transpose(2, 0, 1).shape
                )
        paddle.disable_static()


if __name__ == '__main__':
    unittest.main()
