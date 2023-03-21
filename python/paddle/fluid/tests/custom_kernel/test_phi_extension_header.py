# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import re
import sys
import tempfile
import unittest

import paddle


class TestCustomCPUPlugin(unittest.TestCase):
    def setUp(self):
        paddle_install_dir = re.compile('/__init__.py.*').sub(
            '', paddle.__file__
        )
        paddle_include_dir = os.path.join(paddle_install_dir, 'include')

        paddle_binary_dir = os.path.join(
            os.environ['PADDLE_BINARY_DIR'], 'third_party'
        )
        third_include_glog = os.path.join(
            paddle_binary_dir, 'install/glog/include'
        )
        third_include_gflags = os.path.join(
            paddle_binary_dir, 'install/gflags/include'
        )

        self.temp_dir = tempfile.TemporaryDirectory()
        with open(os.path.join(self.temp_dir.name, "test.cc"), "w") as fd:
            fd.write('#include "paddle/phi/extension.h"\n')
            fd.write('int main() { return 0; }\n')

        self.cmd = 'cd {} && g++ -DPADDLE_WITH_CUSTOM_KERNEL -I{} -I{} -I{} -O3 test.cc'.format(
            self.temp_dir.name,
            paddle_include_dir,
            third_include_glog,
            third_include_gflags,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_header_compile(self):
        exitcode = os.system(self.cmd)
        assert exitcode == 0


if __name__ == '__main__':
    if os.name == 'nt' or sys.platform.startswith('darwin'):
        # only support Linux now
        exit()
    unittest.main()
