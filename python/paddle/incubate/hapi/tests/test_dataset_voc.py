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
import os
import numpy as np
import tempfile
import shutil
import cv2

from paddle.incubate.hapi.datasets import voc2012, VOC2012
from paddle.incubate.hapi.datasets.utils import _check_exists_and_download

# VOC2012 is too large for unittest to download, stub a small dataset here
voc2012.VOC_URL = 'https://paddlemodels.bj.bcebos.com/voc2012_stub/VOCtrainval_11-May-2012.tar'
voc2012.VOC_MD5 = '34cb1fe5bdc139a5454b25b16118fff8'


class TestVOC2012Train(unittest.TestCase):
    def test_main(self):
        voc2012 = VOC2012(mode='train')
        self.assertTrue(len(voc2012) == 3)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 3)
        image, label = voc2012[idx]
        self.assertTrue(len(image.shape) == 3)
        self.assertTrue(len(label.shape) == 2)


class TestVOC2012Valid(unittest.TestCase):
    def test_main(self):
        voc2012 = VOC2012(mode='valid')
        self.assertTrue(len(voc2012) == 1)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 1)
        image, label = voc2012[idx]
        self.assertTrue(len(image.shape) == 3)
        self.assertTrue(len(label.shape) == 2)


class TestVOC2012Test(unittest.TestCase):
    def test_main(self):
        voc2012 = VOC2012(mode='test')
        self.assertTrue(len(voc2012) == 2)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 1)
        image, label = voc2012[idx]
        self.assertTrue(len(image.shape) == 3)
        self.assertTrue(len(label.shape) == 2)


if __name__ == '__main__':
    unittest.main()
