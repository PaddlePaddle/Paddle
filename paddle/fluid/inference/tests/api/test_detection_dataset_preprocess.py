# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

from local_detection_set_preprocess import pascalvoc
from full_pascalvoc_test_preprocess import run_convert
import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid
import unittest
import os


class Test_Preprocess(unittest.TestCase):
    def test_local_convert(self):
        DATA_DIR = "~/data/pascalvoc/"
        FILE_LIST = "test_head_100.txt"
        label_file = "label_list"
        IMAGE_OUT = 'pascalvoc_val_head_100.bin'
        DATA_DIR = os.path.expanduser(DATA_DIR)
        pascalvoc(DATA_DIR, IMAGE_OUT, label_file, FILE_LIST)

    def test_online_convert(self):
        run_convert()


if __name__ == '__main__':
    unittest.main()
