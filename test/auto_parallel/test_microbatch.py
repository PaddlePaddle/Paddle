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
from paddle.distributed.auto_parallel.pipelining.microbatch import (
    TensorChunkSpec,
    merge_chunks,
    split_args_kwargs_into_chunks,
)


class TestMicrobatch(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.batch_size = 8
        self.feature_size = 4
        self.tensor = paddle.randn([self.batch_size, self.feature_size])

    def test_tensor_chunk_spec(self):
        # Test creation and string representation of TensorChunkSpec
        spec = TensorChunkSpec(0)
        self.assertEqual(spec.split_axis, 0)
        self.assertEqual(str(spec), "TensorChunkSpec(0)")
        self.assertTrue("TensorChunkSpec(0)" in repr(spec))

    def test_split_args_kwargs(self):
        # Test basic parameter splitting
        args = (self.tensor,)
        kwargs = {"input": self.tensor}
        num_chunks = 2

        args_split, kwargs_split = split_args_kwargs_into_chunks(
            args, kwargs, num_chunks
        )

        self.assertEqual(len(args_split), num_chunks)
        self.assertEqual(len(kwargs_split), num_chunks)
        self.assertEqual(
            args_split[0][0].shape[0], self.batch_size // num_chunks
        )

        # Test splitting with non-tensor parameters
        args = (self.tensor, 42, "string")
        kwargs = {"tensor": self.tensor, "number": 42}
        num_chunks = 2

        args_split, kwargs_split = split_args_kwargs_into_chunks(
            args, kwargs, num_chunks
        )

        # Verify non-tensor parameters remain unchanged in each chunk
        self.assertEqual(args_split[0][1], 42)
        self.assertEqual(args_split[0][2], "string")
        self.assertEqual(kwargs_split[0]["number"], 42)

        # Test splitting with custom specification
        tensor_2d = paddle.randn([4, 6])
        args = (tensor_2d,)
        args_chunk_spec = (TensorChunkSpec(1),)  # Split on second dimension

        args_split, _ = split_args_kwargs_into_chunks(
            args, None, 2, args_chunk_spec
        )

        self.assertEqual(args_split[0][0].shape[1], 3)

    def test_merge_chunks(self):
        # Test merging chunks
        chunk1 = paddle.randn([4, 4])
        chunk2 = paddle.randn([4, 4])
        chunks = [chunk1, chunk2]
        chunk_spec = [TensorChunkSpec(0)]

        merged = merge_chunks(chunks, chunk_spec)
        self.assertEqual(merged.shape[0], 8)

        # Test merging chunks containing non-tensor values
        chunks = [(paddle.randn([4, 4]), 42)] * 2
        chunk_spec = [TensorChunkSpec(0), None]

        merged = merge_chunks(chunks, chunk_spec)
        self.assertEqual(merged[1], 42)

        # Test error cases
        with self.assertRaises(ValueError):
            # Test error when tensor size is smaller than number of chunks
            small_tensor = paddle.randn([1, 4])
            split_args_kwargs_into_chunks((small_tensor,), None, 2)

        with self.assertRaises(AssertionError):
            # Test error when parameter count doesn't match chunk_spec length
            split_args_kwargs_into_chunks(
                (self.tensor,),
                None,
                2,
                (TensorChunkSpec(0), TensorChunkSpec(1)),
            )

        # test merge empty chunks
        empty_chunks = []
        result = merge_chunks(empty_chunks, None)
        self.assertEqual(result, [])

        # test tensor size smaller than chunks number
        small_tensor = paddle.randn([1, 4])
        with self.assertRaises(ValueError):
            split_args_kwargs_into_chunks((small_tensor,), None, 2)

        # test merge non-tensor with tensor spec
        chunks = [(42,), (42,)]
        chunk_spec = (TensorChunkSpec(0),)
        result = merge_chunks(chunks, chunk_spec)
        self.assertEqual(result[0], 42)

    def test_nested_structure(self):
        # test nested tensor
        nested_tensor = [
            [paddle.randn([4, 2]), paddle.randn([4, 2])],
            [paddle.randn([4, 2]), paddle.randn([4, 2])],
        ]

        args = (nested_tensor,)
        kwargs = {"nested": nested_tensor}

        args_split, kwargs_split = split_args_kwargs_into_chunks(
            args, kwargs, 2
        )

        self.assertEqual(len(args_split), 2)
        self.assertEqual(len(args_split[0][0]), 2)
        self.assertEqual(len(args_split[0][0][0]), 2)
        self.assertEqual(args_split[0][0][0][0].shape, [2, 2])

        self.assertEqual(len(kwargs_split), 2)
        self.assertEqual(len(kwargs_split[0]["nested"]), 2)
        self.assertEqual(len(kwargs_split[0]["nested"][0]), 2)
        self.assertEqual(kwargs_split[0]["nested"][0][0].shape, [2, 2])

        merged_args = merge_chunks(
            args_split,
            [
                [TensorChunkSpec(0), TensorChunkSpec(0)],
                [TensorChunkSpec(0), TensorChunkSpec(0)],
            ],
        )

        self.assertEqual(merged_args[0][0][0].shape, [4, 2])
        self.assertEqual(merged_args[0][1][1].shape, [4, 2])

        self.assertEqual(len(merged_args[0]), 2)
        self.assertEqual(len(merged_args[0][0]), 2)


if __name__ == "__main__":
    unittest.main()
