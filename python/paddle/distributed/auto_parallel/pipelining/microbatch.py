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

from __future__ import annotations

import logging
from typing import Any

import paddle
import paddle.distributed as dist
from paddle.distributed import Replicate, Shard
from paddle.distributed.auto_parallel.api import (
    dtensor_from_local,
    dtensor_to_local,
)
from paddle.utils import flatten, map_structure, pack_sequence_as

logger = logging.getLogger(__name__)

# Default chunking dimension is 0. This is used for the case where the user did
# not specify a chunking dimension.
DEFAULT_CHUNK_DIM = 0


def _split_tensor(x, num_chunks, split_axis=0):
    if not x.is_dist():
        chunk_tensors = paddle.tensor_split(x, num_chunks, split_axis)
    # dp_degree > 1 , placements of model input is [S(0), R, ...]
    else:
        if dist.in_auto_parallel_align_mode():

            def _reorder_data_for_align():
                nonlocal x
                assert x.placements[0] == dist.Shard(
                    0
                ), "inputs should be placed on S(0)."

                shardings = x.process_mesh.shape[0]

                rows_per_shard = x.shape[0] // shardings
                new_indices = []
                for s_id in range(shardings):
                    for row_in_shard in range(rows_per_shard):
                        new_indices.append(s_id + row_in_shard * shardings)
                tmp = x[new_indices]
                x = dist.reshard(tmp, x.process_mesh, x.placements)

            _reorder_data_for_align()
        mesh = x.process_mesh
        placements = x.placements
        dense_x = dtensor_to_local(x, mesh, placements)
        chunk_tensors = paddle.tensor_split(dense_x, num_chunks, split_axis)
        for i in range(num_chunks):
            chunk_tensors[i] = dtensor_from_local(
                chunk_tensors[i], mesh, placements
            )
    return chunk_tensors


def _concat_tensor(chunk_tensors, axis=0):
    chunk0 = chunk_tensors[0]
    if not chunk0.is_dist():
        out = paddle.concat(chunk_tensors, axis)

    else:
        # loss_fun(out, labels), placements of labels is [S(0), R, ...]
        mesh = chunk0.process_mesh
        placements = [Replicate() for _ in range(mesh.ndim)]
        dp_index = mesh.dim_names.index("dp") if "dp" in mesh.dim_names else 0
        placements[dp_index] = Shard(0)

        for i in range(len(chunk_tensors)):
            chunk_tensors[i] = dist.reshard(chunk_tensors[i], mesh, placements)

            chunk_tensors[i] = dtensor_to_local(
                chunk_tensors[i], mesh, placements
            )
        out = paddle.concat(chunk_tensors, axis)
        out = dtensor_from_local(out, mesh, placements)
    return out


class TensorChunkSpec:
    """
    Class used to specify chunking of inputs
    """

    def __init__(self, split_axis):
        self.split_axis = split_axis

    split_axis: int

    def __repr__(self):
        return f"{self.__class__.__module__}.{self.__class__.__name__}({self.split_axis})"

    def __str__(self):
        return f"TensorChunkSpec({self.split_axis})"


def _split_args_helper(
    args_dict,
    args_chunk_spec,
    num_chunks,
):
    """
    A helper function of split_args_kwargs_into_chunks.
    """
    assert len(args_dict) == len(
        args_chunk_spec
    ), f"args_dict.keys() = {list(args_dict.keys())} args_chunk_spec.keys() = {list(args_chunk_spec.keys())}"

    shared_args_dict_flat = {}
    # handle args one by one
    for arg_key, arg in args_dict.items():
        arg_flat = flatten(arg)

        chunk_spec = args_chunk_spec[arg_key]
        assert chunk_spec is not None

        chunk_spec_flat = flatten(chunk_spec)
        assert len(chunk_spec_flat) == len(
            arg_flat
        ), f"{arg_key} {len(arg_flat)} != {len(chunk_spec_flat)}"

        shard_arg_flat = []

        for v, chunk_v in zip(arg_flat, chunk_spec_flat):
            if not isinstance(v, paddle.Tensor):
                shard_arg_flat.append([v] * num_chunks)
            elif isinstance(chunk_v, TensorChunkSpec):
                v_split_axis_size = v.shape[chunk_v.split_axis]

                if v_split_axis_size < num_chunks:
                    raise ValueError(
                        f"Arg {arg_key} on chunking dimension has a size of {v_split_axis_size}, "
                        f"smaller than the number of chunks {num_chunks}. "
                        "Please adjust your num_chunks setting."
                    )
                # split tensor v
                chunk_tensors = _split_tensor(v, num_chunks, chunk_v.split_axis)

                shard_arg_flat.append(chunk_tensors)
            else:
                raise TypeError(f"Unrecognized chunk spec: {chunk_v}")

        shared_args_dict_flat[arg_key] = shard_arg_flat

    # the structure of each element in args_split is the same as the original args_dict
    args_split = []
    for idx in range(num_chunks):
        chunk_args = {}
        for key, arg in shared_args_dict_flat.items():
            last_arg = None if not arg else arg[0][idx]
            arg_of_curr_chunk = (
                [v[idx] for v in arg] if len(arg) > 1 else last_arg
            )
            chunk_args[key] = arg_of_curr_chunk

        # flatten chunk_args first, and then pack chunk_args as the origin args_dict
        flatten_chunk_args = [x for x in flatten(chunk_args) if x is not None]
        chunk_args = pack_sequence_as(args_dict, flatten_chunk_args)
        args_split.append(chunk_args)
    return args_split


def split_args_kwargs_into_chunks(
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
    chunks: int,
    args_chunk_spec: (
        tuple[
            tuple[TensorChunkSpec, ...]
            | list[TensorChunkSpec, ...]
            | TensorChunkSpec,
            ...,
        ]
        | None
    ) = None,
    kwargs_chunk_spec: (
        dict[
            str,
            tuple[TensorChunkSpec, ...]
            | list[TensorChunkSpec, ...]
            | TensorChunkSpec,
        ]
        | None
    ) = None,
) -> tuple[list[tuple], list[dict]]:
    """
    Given a sequence of args and kwargs, split them into a number of chunks
    according to  their respective chunking specs.

    Args:
        args: tuple of args
        kwargs: dict of kwargs
        chunks: Number of chunks to split the args and kwargs into
        args_chunk_spec: chunking specs for args, in same shape as args
        kwargs_chunk_spec: chunking specs for kwargs, in same shape as kwargs

    Returns:
        args_split: list of sharded args
        kwargs_split: list of sharded kwargs
    """

    if kwargs is None:
        kwargs = {}

    if args_chunk_spec is None:
        args_chunk_spec = map_structure(
            lambda _: TensorChunkSpec(DEFAULT_CHUNK_DIM), args
        )

    if kwargs_chunk_spec is None:
        kwargs_chunk_spec = map_structure(
            lambda _: TensorChunkSpec(DEFAULT_CHUNK_DIM), kwargs
        )

    args_split_dict = _split_args_helper(
        dict(enumerate(args)),
        dict(enumerate(args_chunk_spec)),
        chunks,
    )
    kwargs_split = _split_args_helper(
        kwargs,
        kwargs_chunk_spec,
        chunks,
    )

    assert len(args_split_dict) == len(kwargs_split), (
        "args and kwargs are split into difference number of chunks: "
        f"{len(args_split_dict)}, {len(kwargs_split)}"
    )

    # the form of each args_chunk should be tuple
    args_split = [
        tuple(args_chunk[i] for i in range(len(args_chunk)))
        for args_chunk in args_split_dict
    ]

    return args_split, kwargs_split


def merge_chunks(
    chunks: list[Any],
    chunk_spec,
):
    """
    Given a list of chunks, merge them into a single chunk according to
    the chunk spec.

    Args:
        chunks: list of chunks
        chunk_spec: Chunking spec for the chunks

    Returns:
        chunk: chunks merged value
    """
    if len(chunks) == 0:
        logger.warning("No chunks to merge.")
        return chunks

    if chunk_spec is None:
        chunk_spec = map_structure(
            lambda _: TensorChunkSpec(DEFAULT_CHUNK_DIM), chunks[0]
        )

    chunks_flat = []
    # flatten chunk_spec first
    chunk_spec = flatten(chunk_spec)
    for chunk in chunks:
        chunk_flat = flatten(chunk)
        assert len(chunk_flat) == len(
            chunk_spec
        ), f"Chunk {chunk} did not match chunk spec {chunk_spec}"
        chunks_flat.append(chunk_flat)

    def _merge_non_tensor_type_arg(chunks, idx, chunk_spec_of_arg=None):
        # use the first chunk's value as the merged result
        arg_0 = chunks[0][idx]
        for chunk_idx in range(1, len(chunks)):
            assert chunks[chunk_idx][idx] == arg_0, (
                f"Cannot merge chunks with index 0 and {idx} with different values,"
                f"When the arg's TensorChunkSpec is {chunk_spec_of_arg}"
            )
        return arg_0

    args_flat = []
    for arg_idx, chunk_spec_of_arg in enumerate(chunk_spec):
        if isinstance(chunk_spec_of_arg, TensorChunkSpec):
            if isinstance(chunks_flat[0][arg_idx], paddle.Tensor):
                arg_chunks_to_merge = [
                    chunks_flat[chunk_idx][arg_idx]
                    for chunk_idx in range(len(chunks_flat))
                ]
                merged_arg = _concat_tensor(
                    arg_chunks_to_merge, axis=chunk_spec_of_arg.split_axis
                )
            else:
                logger.warning(
                    f"Cannot merge chunks with TensorChunkSpec {chunk_spec_of_arg}."
                    "The TensorChunkSpec only supports paddle.Tensor type."
                )

                merged_arg = _merge_non_tensor_type_arg(
                    chunks_flat, arg_idx, chunk_spec_of_arg
                )
        else:
            merged_arg = _merge_non_tensor_type_arg(
                chunks_flat, arg_idx, chunk_spec_of_arg
            )

        args_flat.append(merged_arg)

    # pack args_flat as the input chunks[0]
    return pack_sequence_as(chunks[0], args_flat)
