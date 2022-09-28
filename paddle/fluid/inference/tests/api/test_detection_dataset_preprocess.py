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

from full_pascalvoc_test_preprocess import main_pascalvoc_preprocess
import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid
import unittest
import os


class Test_Preprocess(unittest.TestCase):

    def test_local_convert(self):
        os.system("python full_pascalvoc_test_preprocess.py --choice=local")

    def test_online_convert(self):
        os.system(
            "python full_pascalvoc_test_preprocess.py --choice=VOC_test_2007")


if __name__ == '__main__':
    unittest.main()
