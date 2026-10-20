#   Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import importlib
import unittest
from unittest.mock import patch

import paddle
from paddle.compat.distributed import all_to_all_single


class TestAllToAllSingle(unittest.TestCase):
    """``paddle.compat.distributed.all_to_all_single`` mirrors torch's API.

    The forwarding is checked against a stub so no process group is needed.
    """

    def setUp(self):
        self.calls = []

        def fake_alltoall_single(
            out_tensor,
            in_tensor,
            in_split_sizes=None,
            out_split_sizes=None,
            group=None,
            sync_op=True,
        ):
            self.calls.append(
                {
                    "out_tensor": out_tensor,
                    "in_tensor": in_tensor,
                    "in_split_sizes": in_split_sizes,
                    "out_split_sizes": out_split_sizes,
                    "group": group,
                    "sync_op": sync_op,
                }
            )
            return "task"

        patcher = patch(
            "paddle.distributed.alltoall_single", fake_alltoall_single
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        self.output = paddle.empty([2], dtype="int64")
        self.input = paddle.arange(4, dtype="int64")

    def test_split_sizes_are_swapped(self):
        # torch passes (output_split_sizes, input_split_sizes) while
        # paddle.distributed.alltoall_single expects (in_split_sizes,
        # out_split_sizes).
        all_to_all_single(self.output, self.input, [1, 1], [2, 2])
        call = self.calls[0]
        self.assertIs(call["out_tensor"], self.output)
        self.assertIs(call["in_tensor"], self.input)
        self.assertEqual(call["in_split_sizes"], [2, 2])
        self.assertEqual(call["out_split_sizes"], [1, 1])

    def test_optional_arguments_default_to_none(self):
        all_to_all_single(self.output, self.input)
        call = self.calls[0]
        self.assertIsNone(call["in_split_sizes"])
        self.assertIsNone(call["out_split_sizes"])
        self.assertIsNone(call["group"])

    def test_group_is_forwarded(self):
        all_to_all_single(self.output, self.input, group="my-group")
        self.assertEqual(self.calls[0]["group"], "my-group")

    def test_async_op_maps_to_sync_op(self):
        # async_op=True means sync_op=False, and the task is handed back.
        self.assertEqual(
            all_to_all_single(self.output, self.input, async_op=True), "task"
        )
        self.assertFalse(self.calls[0]["sync_op"])

        # async_op=False means sync_op=True, and nothing is returned.
        self.assertIsNone(all_to_all_single(self.output, self.input))
        self.assertTrue(self.calls[1]["sync_op"])


class TestCompatDistributedLazyImport(unittest.TestCase):
    def test_module_is_lazily_exposed(self):
        # Enabling the torch proxy imports every ``paddle.compat`` submodule and
        # caches it on the package, which hides the ``__getattr__`` fallback
        # that plain attribute access relies on. Drop the cached attribute so
        # the fallback is exercised whatever order the tests run in.
        paddle.compat.__dict__.pop("distributed", None)
        module = paddle.compat.distributed
        self.assertEqual(module.__name__, "paddle.compat.distributed")
        self.assertIs(
            module, importlib.import_module("paddle.compat.distributed")
        )
        # The fallback caches the module for later accesses.
        self.assertIs(paddle.compat.__dict__["distributed"], module)


if __name__ == '__main__':
    unittest.main()
