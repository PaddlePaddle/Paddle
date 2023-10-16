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

from paddle.base import core


class TestTF32Switch(unittest.TestCase):
    def test_on_off(self):
        if core.is_compiled_with_cuda():
            self.assertTrue(core.get_cudnn_switch())  # default
            core.set_cudnn_switch(0)
            self.assertFalse(core.get_cudnn_switch())  # turn off
            core.set_cudnn_switch(1)
            self.assertTrue(core.get_cudnn_switch())  # turn on

            core.set_cudnn_switch(1)  # restore the switch
        else:
            pass


if __name__ == '__main__':
    unittest.main()
