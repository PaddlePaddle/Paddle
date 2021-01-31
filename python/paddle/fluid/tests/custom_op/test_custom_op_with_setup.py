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

import os
import unittest

from test_custom_op import CustomOpTest, load_so


def compile_so():
    """
    Compile .so file by running setup.py config.
    """
    # build .so with setup.py
    file_dir = os.path.dirname(os.path.abspath(__file__))
    os.system('cd {} && python setup.py build'.format(file_dir))


if __name__ == '__main__':
    compile_so()
    load_so(so_name='librelu2_op_from_setup.so')
    unittest.main()
