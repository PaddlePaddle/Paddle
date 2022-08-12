#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle


class TestCPUVersion(unittest.TestCase):

    def test_cuda_cudnn_version_in_cpu_package(self):
        if not paddle.is_compiled_with_cuda():
            self.assertEqual(paddle.version.cuda(), 'False')
            self.assertEqual(paddle.version.cudnn(), 'False')


if __name__ == '__main__':
    unittest.main()
