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

from paddle.jit.sot.opcode_translator.executor.variable_stack import (
    VariableStack,
)


class TestVariableStack(unittest.TestCase):
    def test_basic(self):
        stack = VariableStack([1, 2, 3])
        self.assertEqual(str(stack), "[1, 2, 3]")
        self.assertEqual(len(stack), 3)
        self.assertEqual(str(stack.copy()), str(stack))

    def test_peek(self):
        stack = VariableStack([1, 2, 3])
        self.assertEqual(stack.peek(), 3)
        self.assertEqual(stack.top, 3)
        self.assertEqual(stack.peek(1), 3)
        stack.peek[1] = 4
        stack.peek[2] = 3
        self.assertEqual(stack.peek[1], 4)
        self.assertEqual(stack.peek[:1], [4])
        self.assertEqual(stack.peek[:2], [3, 4])
        stack.top = 5
        self.assertEqual(stack.peek[:2], [3, 5])

    def test_push_pop(self):
        stack = VariableStack()
        stack.push(1)
        stack.push(2)
        self.assertEqual(stack.pop(), 2)
        self.assertEqual(stack.pop(), 1)

    def test_pop_n(self):
        stack = VariableStack([1, 2, 3, 4])
        self.assertEqual(stack.pop_n(2), [3, 4])
        self.assertEqual(stack.pop_n(2), [1, 2])


if __name__ == "__main__":
    unittest.main()
