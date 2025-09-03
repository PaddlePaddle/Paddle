# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


@paddle.library.custom_op(
    "test_namespace::add_one",
    mutates_args=(),
)
def add_one(x):
    return x + 1


@add_one.register_fake
def add_one_fake_fn(x):
    return x


@paddle.library.custom_op(
    "test_namespace::add_two",
    mutates_args=(),
)
def add_two(x):
    return x + 2


class TestCallCustomOp(unittest.TestCase):
    def test_call_custom_op(self):
        self.assertEqual(paddle.ops.test_namespace.add_one(1), 2)


class TestRegisterFake(unittest.TestCase):
    def test_register_fake(self):
        paddle.library.register_fake(
            "test_namespace::add_two",
            lambda x: x,
        )


if __name__ == "__main__":
    unittest.main()
