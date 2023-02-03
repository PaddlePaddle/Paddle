#!/bin/bash

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

import os
import shlex  # noqa: F401
import unittest


class GpuBoxTest(unittest.TestCase):
    def test_gpubox(self):
        exitcode = os.system('sh gpubox_run.sh')
        os.system('rm *_train_desc.prototxt')
        if os.path.exists('./train_data'):
            os.system('rm -rf train_data')
        if os.path.exists('./log'):
            os.system('rm -rf log')


if __name__ == '__main__':
    if not os.path.exists('./train_data'):
        os.system('sh download_criteo_data.sh')
    unittest.main()
