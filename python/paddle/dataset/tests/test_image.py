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

import unittest
import numpy as np

import paddle.dataset.image as image

__all__ = []


class Image(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()
