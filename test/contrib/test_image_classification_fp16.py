#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import contextlib
import copy
import sys
import tempfile
import unittest

# TODO: remove sys.path.append
sys.path.append("../../legacy_test")

import paddle
from paddle import base

paddle.enable_static()


class TestImageClassification(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_amp_lists(self):
        white_list = (
            copy.copy(paddle.static.amp.fp16_lists.white_list)
            | paddle.static.amp.fp16_lists._only_supported_fp16_list
        )
        black_list = copy.copy(
            paddle.static.amp.fp16_lists.black_list
            | paddle.static.amp.fp16_lists._extra_black_list
        )
        gray_list = copy.copy(paddle.static.amp.fp16_lists.gray_list)

        amp_lists = paddle.static.amp.AutoMixedPrecisionLists()
        self.assertEqual(amp_lists.white_list, white_list)
        self.assertEqual(amp_lists.black_list, black_list)
        self.assertEqual(amp_lists.gray_list, gray_list)

    def test_amp_lists_1(self):
        white_list = (
            copy.copy(paddle.static.amp.fp16_lists.white_list)
            | paddle.static.amp.fp16_lists._only_supported_fp16_list
        )
        black_list = copy.copy(
            paddle.static.amp.fp16_lists.black_list
            | paddle.static.amp.fp16_lists._extra_black_list
        )
        gray_list = copy.copy(paddle.static.amp.fp16_lists.gray_list)

        # 1. w={'exp}, b=None
        white_list.add('exp')
        black_list.remove('exp')

        amp_lists = paddle.static.amp.AutoMixedPrecisionLists({'exp'})
        self.assertEqual(amp_lists.white_list, white_list)
        self.assertEqual(amp_lists.black_list, black_list)
        self.assertEqual(amp_lists.gray_list, gray_list)

    def test_amp_lists_2(self):
        white_list = (
            copy.copy(paddle.static.amp.fp16_lists.white_list)
            | paddle.static.amp.fp16_lists._only_supported_fp16_list
        )
        black_list = copy.copy(
            paddle.static.amp.fp16_lists.black_list
            | paddle.static.amp.fp16_lists._extra_black_list
        )
        gray_list = copy.copy(paddle.static.amp.fp16_lists.gray_list)

        # 2. w={'tanh'}, b=None
        white_list.add('tanh')
        gray_list.remove('tanh')

        amp_lists = paddle.static.amp.AutoMixedPrecisionLists({'tanh'})
        self.assertEqual(amp_lists.white_list, white_list)
        self.assertEqual(amp_lists.black_list, black_list)
        self.assertEqual(amp_lists.gray_list, gray_list)

    def test_amp_lists_3(self):
        white_list = (
            copy.copy(paddle.static.amp.fp16_lists.white_list)
            | paddle.static.amp.fp16_lists._only_supported_fp16_list
        )
        black_list = copy.copy(
            paddle.static.amp.fp16_lists.black_list
            | paddle.static.amp.fp16_lists._extra_black_list
        )
        gray_list = copy.copy(paddle.static.amp.fp16_lists.gray_list)

        # 3. w={'lstm'}, b=None
        white_list.add('lstm')

        amp_lists = paddle.static.amp.AutoMixedPrecisionLists({'lstm'})
        self.assertEqual(amp_lists.white_list, white_list)
        self.assertEqual(amp_lists.black_list, black_list)
        self.assertEqual(amp_lists.gray_list, gray_list)

    def test_amp_lists_4(self):
        white_list = (
            copy.copy(paddle.static.amp.fp16_lists.white_list)
            | paddle.static.amp.fp16_lists._only_supported_fp16_list
        )
        black_list = copy.copy(
            paddle.static.amp.fp16_lists.black_list
            | paddle.static.amp.fp16_lists._extra_black_list
        )
        gray_list = copy.copy(paddle.static.amp.fp16_lists.gray_list)

        # 4. w=None, b={'conv2d'}
        white_list.remove('conv2d')
        black_list.add('conv2d')

        amp_lists = paddle.static.amp.AutoMixedPrecisionLists(
            custom_black_list={'conv2d'}
        )
        self.assertEqual(amp_lists.white_list, white_list)
        self.assertEqual(amp_lists.black_list, black_list)
        self.assertEqual(amp_lists.gray_list, gray_list)

    def test_amp_lists_5(self):
        white_list = (
            copy.copy(paddle.static.amp.fp16_lists.white_list)
            | paddle.static.amp.fp16_lists._only_supported_fp16_list
        )
        black_list = copy.copy(
            paddle.static.amp.fp16_lists.black_list
            | paddle.static.amp.fp16_lists._extra_black_list
        )
        gray_list = copy.copy(paddle.static.amp.fp16_lists.gray_list)

        # 5. w=None, b={'tanh'}
        black_list.add('tanh')
        gray_list.remove('tanh')

        amp_lists = paddle.static.amp.AutoMixedPrecisionLists(
            custom_black_list={'tanh'}
        )
        self.assertEqual(amp_lists.white_list, white_list)
        self.assertEqual(amp_lists.black_list, black_list)
        self.assertEqual(amp_lists.gray_list, gray_list)

    def test_amp_lists_6(self):
        white_list = (
            copy.copy(paddle.static.amp.fp16_lists.white_list)
            | paddle.static.amp.fp16_lists._only_supported_fp16_list
        )
        black_list = copy.copy(
            paddle.static.amp.fp16_lists.black_list
            | paddle.static.amp.fp16_lists._extra_black_list
        )
        gray_list = copy.copy(paddle.static.amp.fp16_lists.gray_list)

        # 6. w=None, b={'lstm'}
        black_list.add('lstm')

        amp_lists = paddle.static.amp.AutoMixedPrecisionLists(
            custom_black_list={'lstm'}
        )
        self.assertEqual(amp_lists.white_list, white_list)
        self.assertEqual(amp_lists.black_list, black_list)
        self.assertEqual(amp_lists.gray_list, gray_list)

    def test_amp_lists_7(self):
        # 7. w={'lstm'} b={'lstm'}
        # raise ValueError
        self.assertRaises(
            ValueError,
            paddle.static.amp.AutoMixedPrecisionLists,
            {'lstm'},
            {'lstm'},
        )

    @contextlib.contextmanager
    def scope_prog_guard(self):
        prog = base.Program()
        startup_prog = base.Program()
        scope = base.core.Scope()
        with base.scope_guard(scope):
            with base.program_guard(prog, startup_prog):
                yield


if __name__ == '__main__':
    unittest.main()
