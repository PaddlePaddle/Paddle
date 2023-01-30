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
<<<<<<< HEAD
"""
Fliename:
    test_image.py
Description:
    This scipt test image resize,flip and chw.
"""

import sys
import unittest

import numpy as np

from paddle.dataset import image
=======

from __future__ import print_function

import unittest
import numpy as np

import paddle.dataset.image as image
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

__all__ = []


class Image(unittest.TestCase):
<<<<<<< HEAD
    """
    This function is test image resize,flip and chw.
    """

    def test_resize_flip_chw(self):
        """resize"""
        imgdir = sys.argv[0].replace('test_image.py', 'cat.jpg')
        images = image.load_image(imgdir)
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
=======

    def test_resize_flip_chw(self):
        # resize
        im = image.load_image('cat.jpg')
        im = image.resize_short(im, 256)
        self.assertEqual(256, min(im.shape[:2]))
        self.assertEqual(3, im.shape[2])

        # flip
        im = image.left_right_flip(im)
        im2 = np.flip(im, 1)
        self.assertEqual(im.all(), im2.all())

        # to_chw
        h, w, c = im.shape
        im = image.to_chw(im)
        self.assertEqual(c, im.shape[0])
        self.assertEqual(h, im.shape[1])
        self.assertEqual(w, im.shape[2])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == '__main__':
    unittest.main()
