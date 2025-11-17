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

import unittest

import paddle


class TestAPI(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()

    def assert_api(self, api_func, require_grad):
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = api_func()
        self.assertEqual(x.stop_gradient, require_grad)
        # test for setter
        x.requires_grad_(require_grad)
        self.assertEqual(x.stop_gradient, not require_grad)

    def test_full(self):
        api = lambda: paddle.full(shape=[2, 3], fill_value=1.0)
        self.assert_api(api, True)

    def test_data(self):
        api = lambda: paddle.static.data('x', [4, 4], dtype='float32')
        self.assert_api(api, True)

    # TODO(Aurelius84): Add more test cases after API is migrated.


class TestParameters(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()

    def test_create_param(self):
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            w = paddle.create_parameter(shape=[784, 200], dtype='float32')
        self.assertEqual(w.stop_gradient, False)
        self.assertEqual(w.persistable, True)

        # test for setter
        w.requires_grad_(False)
        w.persistable = False
        self.assertEqual(w.stop_gradient, True)
        self.assertEqual(w.persistable, False)


if __name__ == '__main__':
    unittest.main()
