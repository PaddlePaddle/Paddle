# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.incubate.hapi.vision.transforms import transforms


class TestTransforms(unittest.TestCase):
    def do_transform(self, trans):
        fake_img = (np.random.random((400, 300, 3)) * 255).astype('uint8')
        for t in trans:
            fake_img = t(fake_img)

    def test_color_jitter(self):
        trans = [
            transforms.BrightnessTransform(0.0), transforms.HueTransform(0.0),
            transforms.SaturationTransform(0.0),
            transforms.ContrastTransform(0.0),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)
        ]
        self.do_transform(trans)

    def test_exception(self):

        with self.assertRaises(ValueError):
            transforms.ContrastTransform(-1.0)

        with self.assertRaises(ValueError):
            transforms.SaturationTransform(-1.0),

        with self.assertRaises(ValueError):
            transforms.HueTransform(-1.0)

        with self.assertRaises(ValueError):
            transforms.BrightnessTransform(-1.0)


if __name__ == '__main__':
    unittest.main()
