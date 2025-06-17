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

import paddle
from paddle.distributed.auto_parallel.pipelining.microbatch import (
    TensorChunkSpec,
    merge_chunks,
    split_args_kwargs_into_chunks,
)


class TestMicrobatch:
    def __init__(self):
        paddle.seed(2024)
        paddle.distributed.init_parallel_env()
        self.batch_size = 8
        self.feature_size = 4
        self.tensor = paddle.randn([self.batch_size, self.feature_size])
        self.rank = paddle.distributed.get_rank()

    def test_tensor_chunk_spec(self):
        # Test creation and string representation of TensorChunkSpec
        spec = TensorChunkSpec(0)
        assert spec.split_axis == 0
        assert str(spec) == "TensorChunkSpec(0)"
        assert "TensorChunkSpec(0)" in repr(spec)

    def test_split_args_kwargs(self):
        # Test basic parameter splitting
        args = (self.tensor,)
        kwargs = {"input": self.tensor}
        num_chunks = 2

        args_split, kwargs_split = split_args_kwargs_into_chunks(
            args, kwargs, num_chunks
        )

        assert len(args_split) == num_chunks
        assert len(kwargs_split) == num_chunks
        assert args_split[0][0].shape[0] == self.batch_size // num_chunks

        # Test splitting with non-tensor parameters
        args = (self.tensor, 42, "string")
        kwargs = {"tensor": self.tensor, "number": 42}
        num_chunks = 2

        args_split, kwargs_split = split_args_kwargs_into_chunks(
            args, kwargs, num_chunks
        )

        # Verify non-tensor parameters remain unchanged in each chunk
        assert args_split[0][1] == 42
        assert args_split[0][2] == "string"
        assert kwargs_split[0]["number"] == 42

        # Test splitting with custom specification
        tensor_2d = paddle.randn([4, 6])
        args = (tensor_2d,)
        args_chunk_spec = (TensorChunkSpec(1),)  # Split on second dimension

        args_split, _ = split_args_kwargs_into_chunks(
            args, None, 2, args_chunk_spec
        )

        assert args_split[0][0].shape[1] == 3

    def test_merge_chunks(self):
        # Test merging chunks
        chunk1 = paddle.randn([4, 4])
        chunk2 = paddle.randn([4, 4])
        chunks = [chunk1, chunk2]
        chunk_spec = [TensorChunkSpec(0)]

        merged = merge_chunks(chunks, chunk_spec)
        assert merged.shape[0] == 8

        # Test merging chunks containing non-tensor values
        chunks = [(paddle.randn([4, 4]), 42)] * 2
        chunk_spec = [TensorChunkSpec(0), None]

        merged = merge_chunks(chunks, chunk_spec)
        assert merged[1] == 42

        # Test error cases
        try:
            # Test error when tensor size is smaller than number of chunks
            small_tensor = paddle.randn([1, 4])
            split_args_kwargs_into_chunks((small_tensor,), None, 2)
            raise AssertionError("Expected ValueError")
        except ValueError:
            pass

        try:
            # Test error when parameter count doesn't match chunk_spec length
            split_args_kwargs_into_chunks(
                (self.tensor,),
                None,
                2,
                (TensorChunkSpec(0), TensorChunkSpec(1)),
            )
            raise AssertionError("Expected ValueError")
        except AssertionError:
            pass

        # test merge empty chunks
        empty_chunks = []
        result = merge_chunks(empty_chunks, None)
        assert result == []

        # test tensor size smaller than chunks number
        small_tensor = paddle.randn([1, 4])
        try:
            split_args_kwargs_into_chunks((small_tensor,), None, 2)
            raise AssertionError("Expected ValueError")
        except ValueError:
            pass

        # test merge non-tensor with tensor spec
        chunks = [(42,), (42,)]
        chunk_spec = (TensorChunkSpec(0),)
        result = merge_chunks(chunks, chunk_spec)
        assert result[0] == 42

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

        assert len(args_split) == 2
        assert len(args_split[0][0]) == 2
        assert len(args_split[0][0][0]) == 2
        assert args_split[0][0][0][0].shape == [2, 2]

        assert len(kwargs_split) == 2
        assert len(kwargs_split[0]["nested"]) == 2
        assert len(kwargs_split[0]["nested"][0]) == 2
        assert kwargs_split[0]["nested"][0][0].shape == [2, 2]

        merged_args = merge_chunks(
            args_split,
            [
                [TensorChunkSpec(0), TensorChunkSpec(0)],
                [TensorChunkSpec(0), TensorChunkSpec(0)],
            ],
        )

        assert merged_args[0][0][0].shape == [4, 2]
        assert merged_args[0][1][1].shape == [4, 2]

        assert len(merged_args[0]) == 2
        assert len(merged_args[0][0]) == 2

    def test_dist_tensor_split_and_merge(self):
        # test dist tensor split and merge
        base_tensor = self.tensor
        dense_tensor, _ = split_args_kwargs_into_chunks(
            (base_tensor,),
            None,
            2,
        )
        mesh = paddle.distributed.ProcessMesh([0, 1], dim_names=["dp"])
        dist_tensor = paddle.distributed.shard_tensor(
            self.tensor,
            mesh,
            [paddle.distributed.Shard(0)],
        )
        dist_tensor_split, _ = split_args_kwargs_into_chunks(
            (dist_tensor,),
            None,
            2,
        )
        if self.rank == 0:
            is_equal = (
                dist_tensor_split[0][0]
                ._local_value()
                .equal_all(dense_tensor[0][0][:2])
            )
            assert is_equal.item()
            is_equal = (
                dist_tensor_split[1][0]
                ._local_value()
                .equal_all(dense_tensor[0][0][2:])
            )
            assert is_equal.item()
        else:
            is_equal = (
                dist_tensor_split[0][0]
                ._local_value()
                .equal_all(dense_tensor[1][0][:2])
            )
            assert is_equal.item()
            is_equal = (
                dist_tensor_split[1][0]
                ._local_value()
                .equal_all(dense_tensor[1][0][2:])
            )
            assert is_equal.item()
        chunk1 = dist_tensor_split[0][0]
        chunk2 = dist_tensor_split[1][0]
        chunk_spec = [TensorChunkSpec(0)]
        merged_chunk = merge_chunks([chunk1, chunk2], chunk_spec)
        if self.rank == 0:
            is_equal = merged_chunk._local_value().equal_all(base_tensor[:4])
            assert is_equal.item()
        else:
            is_equal = merged_chunk._local_value().equal_all(base_tensor[4:])
            assert is_equal.item()

    def run_all_tests(self):
        """Run all test methods"""
        self.test_tensor_chunk_spec()
        self.test_split_args_kwargs()
        self.test_merge_chunks()
        self.test_nested_structure()
        self.test_dist_tensor_split_and_merge()


if __name__ == "__main__":
    TestMicrobatch().run_all_tests()
