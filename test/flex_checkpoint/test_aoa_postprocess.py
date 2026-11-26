# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
"""Parsing of the postprocess list a slice ref carries."""

import unittest

from paddle.distributed.flex_checkpoint.aoa.postprocess import (
    is_permute,
    is_reshape,
    make_reshape_op,
    reshape_op_shape,
    shape_after_postprocess,
)


class TestEntryKinds(unittest.TestCase):
    def test_reshape_round_trip(self):
        operation = make_reshape_op((2, 3, 4))
        self.assertEqual(operation, "reshape:[2, 3, 4]")
        self.assertTrue(is_reshape(operation))
        self.assertEqual(reshape_op_shape(operation), (2, 3, 4))

    def test_kinds_are_told_apart(self):
        for operation, permute, reshape in (
            ("[1, 0]", True, False),
            ("reshape:[2, 3]", False, True),
            ("float16", False, False),
            (None, False, False),
        ):
            with self.subTest(operation=operation):
                self.assertEqual(is_permute(operation), permute)
                self.assertEqual(is_reshape(operation), reshape)


class TestShapeAfterPostprocess(unittest.TestCase):
    def test_permute_reshape_and_cast(self):
        self.assertEqual(
            shape_after_postprocess(
                (4, 6), ["[1, 0]", "reshape:[2, 3, 4]", "[2, 0, 1]", "float16"]
            ),
            [4, 2, 3],
        )

    def test_several_reshapes(self):
        self.assertEqual(
            shape_after_postprocess(
                (4, 6), ["reshape:[2, 12]", "[1, 0]", "reshape:[3, 8]"]
            ),
            [3, 8],
        )

    def test_reshape_with_another_element_count_is_refused(self):
        with self.assertRaisesRegex(ValueError, "element counts differ"):
            shape_after_postprocess((2, 4), ["reshape:[2, 3, 2, 2]"])


if __name__ == '__main__':
    unittest.main()
