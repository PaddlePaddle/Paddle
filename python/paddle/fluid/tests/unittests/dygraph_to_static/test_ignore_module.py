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

import astor
import scipy

from paddle.jit import ignore_module
from paddle.jit.dy2static.convert_call_func import BUILTIN_LIKELY_MODULES


class TestIgnoreModule(unittest.TestCase):
    def test_ignore_module(self):
        modules = [scipy, astor]
        ignore_module(modules)
        self.assertEquals(
            [scipy, astor],
            BUILTIN_LIKELY_MODULES[-2:],
            'Failed to add modules that ignore transcription',
        )


if __name__ == '__main__':
    unittest.main()
