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
"""
Filename:
    test_image.py
Description:
    This script test image resize,flip and chw.
"""
import os
import unittest

import numpy as np

from paddle.dataset import image

__all__ = []


class Image(unittest.TestCase):
    """
    This function is test image resize,flip and chw.
    """

    def test_resize_flip_chw(self):
        """resize"""
        img_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'cat.jpg'
        )
        images = image.load_image(img_dir)
        images = image.resize_short(images, 256)
        self.assertEqual(256, min(images.shape[:2]))
        self.assertEqual(3, images.shape[2])

        # flip
        images = image.left_right_flip(images)
        images2 = np.flip(images, 1)
        self.assertEqual(images.all(), images2.all())

        # to_chw
        height, width, channel = images.shape
        images = image.to_chw(images)
        self.assertEqual(channel, images.shape[0])
        self.assertEqual(height, images.shape[1])
        self.assertEqual(width, images.shape[2])


if __name__ == '__main__':
    unittest.main()
