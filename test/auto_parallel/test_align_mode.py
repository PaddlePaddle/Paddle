# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


class TestAlignModeFLAGS(unittest.TestCase):
    def _set_and_check_align_mode(self, enable_align_mode):
        paddle.set_flags(
            {'FLAGS_enable_auto_parallel_align_mode': enable_align_mode}
        )
        assert (
            paddle.distributed.in_auto_parallel_align_mode()
            == enable_align_mode
        ), "align mode not set correctly"

    def test_align_mode_flags(self):
        self._set_and_check_align_mode(True)
        self._set_and_check_align_mode(False)


if __name__ == "__main__":
    unittest.main()
