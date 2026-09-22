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
"""Fast path for ``check_resumable_locally``.

The stock check answers "can every rank restore straight from its own
``.distcp``?" by comparing the live ``state_dict`` against ``file_storage_info``,
which is derived from ``Metadata.storage_metadata``. Obtaining that structure
costs a full ``paddle.load`` of the metadata file plus an index build, and on a
fully replicated checkpoint ``storage_metadata`` holds ``num_keys * world_size``
entries -- hundreds of MiB and tens of seconds, to produce a single boolean.

This module answers the same question from two much cheaper sources:

1. ``state_dict_metadata`` alone, obtained through a partial parse of the
   metadata file (:func:`~.metadata_reader.load_state_dict_metadata`). If every
   tensor of the checkpoint is stored whole -- a single shard description, no
   flattening, zero global offset, local shape equal to global shape -- then a
   tensor key identifies its ``LocalTensorIndex`` uniquely, so
   ``storage_metadata`` carries no information the key alone does not.
2. The rank's own ``.distcp`` header
   (:func:`~.distcp_reader.scan_tensor_shapes`), which lists exactly the keys and
   shapes the file really holds.

When condition 1 does not hold, shard identity is genuinely ambiguous: the shards
of a split tensor share a shape and differ only in ``global_offset``, which is
recorded nowhere but ``storage_metadata``. In that case this module declines
(returns ``None``) so the caller can fall back to the stock check.

Two properties make it a drop-in replacement:

* The applicability verdict is a pure function of the shared metadata file, so
  every rank reaches the same decision before any communication happens. Only
  the per-rank file check can differ, and it yields a plain bool that is folded
  into the same ``all_gather_object`` the stock check uses.
* Exactly one ``all_gather_object`` is issued when the check applies and none
  when it declines, which matches the stock collective count on either branch.

The check is never more permissive than the stock one: it validates the file's
real contents instead of what the metadata claims about them, so an unfinished or
truncated ``.distcp`` -- which the stock check accepts, since it only calls
``os.path.isfile`` -- makes this one fall back to resharding. Like the stock
check it compares shapes but not dtypes.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import paddle
from paddle.distributed.fleet.utils.log_util import logger

from .distcp_reader import scan_tensor_shapes
from .metadata_reader import load_state_dict_metadata
from .utils import extract_tensor_metadata

if TYPE_CHECKING:
    from paddle import Tensor
    from paddle.distributed.collective import Group

    from .metadata import LocalTensorMetadata
    from .sharded_weight import ShardedWeight


def is_unsharded_state_dict_metadata(
    state_dict_metadata: dict[str, list[LocalTensorMetadata]],
) -> bool:
    """
    Whether every tensor of the checkpoint is stored as one whole piece.

    A tensor is whole when exactly one shard description exists for it globally
    and that description covers the entire tensor. Under this condition a tensor
    key maps to exactly one ``LocalTensorIndex``, which is what allows the local
    resume check to skip ``storage_metadata`` entirely.
    """
    for metas in state_dict_metadata.values():
        if not isinstance(metas, (list, tuple)):
            metas = [metas]
        # More than one distinct description means the tensor is split.
        if len(metas) != 1:
            return False
        meta = metas[0]
        if meta.is_flattened or meta.flattened_range is not None:
            return False
        if any(offset != 0 for offset in meta.global_offset):
            return False
        local_shape = tuple(meta.local_shape)
        if (
            meta.global_shape is not None
            and tuple(meta.global_shape) != local_shape
        ):
            return False
    return True


def _file_matches_state_dict(
    ckpt_file: str, state_dict: dict[str, Tensor] | dict[str, ShardedWeight]
) -> bool:
    """Whether `ckpt_file` holds every tensor `state_dict` needs, whole."""
    try:
        file_shapes = scan_tensor_shapes(ckpt_file)
    except Exception as e:
        logger.info(
            f"[flex_checkpoint][fast_check] could not read the header of "
            f"'{ckpt_file}' ({type(e).__name__}: {e}), treating the checkpoint "
            "as not resumable locally."
        )
        return False

    for key, value in state_dict.items():
        _, local_tensor_metadata = extract_tensor_metadata(value)
        if local_tensor_metadata is None:
            continue
        shape = file_shapes.get(key)
        if shape is None:
            logger.info(
                f"[flex_checkpoint][fast_check] '{key}' is not stored in "
                f"'{os.path.basename(ckpt_file)}', so this rank cannot resume "
                "from its own file."
            )
            return False
        # An unsharded checkpoint pins the stored index to exactly
        #   (key, all-zero offset, not flattened, no range, whole shape),
        # so the live index is compared against those values field by field:
        # the shape comes from the file, the rest are the constants that
        # ``is_unsharded_state_dict_metadata`` has already guaranteed.
        if (
            local_tensor_metadata.is_flattened
            or local_tensor_metadata.flattened_range is not None
            or any(
                offset != 0 for offset in local_tensor_metadata.global_offset
            )
            or tuple(local_tensor_metadata.local_shape) != shape
        ):
            logger.info(
                f"[flex_checkpoint][fast_check] '{key}' does not match what "
                f"'{os.path.basename(ckpt_file)}' holds: wanted "
                f"local_shape={tuple(local_tensor_metadata.local_shape)}, "
                f"global_offset={tuple(local_tensor_metadata.global_offset)}, "
                f"is_flattened={local_tensor_metadata.is_flattened}; the file "
                f"stores shape={shape}."
            )
            return False
    logger.info(
        f"[flex_checkpoint][fast_check] header of "
        f"'{os.path.basename(ckpt_file)}' lists {len(file_shapes)} tensors and "
        f"covers all {len(state_dict)} keys of the state_dict."
    )
    return True


def check_resumable_locally_fast(
    metadata_path: str,
    path: str,
    state_dict: dict[str, Tensor] | dict[str, ShardedWeight],
    use_dist: bool,
    process_group: Group | None = None,
) -> bool | None:
    """
    Cheap tri-state version of ``check_resumable_locally``.

    Args:
        metadata_path(str): Path of the checkpoint's ``*.metadata`` file.
        path(str): The checkpoint directory.
        state_dict: The state_dict that is about to be loaded.
        use_dist(bool): Whether the load runs in distributed mode.
        process_group: The group used to agree on the verdict across ranks.

    Returns:
        True or False when the fast path applies and has decided, None when it
        declines and the caller must fall back to ``check_resumable_locally``.
    """
    if not os.path.isfile(metadata_path):
        logger.info(
            f"[flex_checkpoint][fast_check] '{metadata_path}' does not exist, "
            "declining in favour of the stock check."
        )
        return None

    start = time.time()
    state_dict_metadata = load_state_dict_metadata(
        metadata_path, allow_full_load=False
    )
    if state_dict_metadata is None:
        # Could only be obtained through a full load, which is exactly what the
        # stock check already does -- leave that to the fallback so the metadata
        # file is never read in full twice.
        logger.info(
            f"[flex_checkpoint][fast_check] '{metadata_path}' could not be "
            "parsed partially, declining in favour of the stock check."
        )
        return None
    logger.info(
        f"[flex_checkpoint][fast_check] read state_dict_metadata of "
        f"{len(state_dict_metadata)} tensors from '{metadata_path}' in "
        f"{time.time() - start:.3f}s without loading storage_metadata."
    )

    if not is_unsharded_state_dict_metadata(state_dict_metadata):
        logger.info(
            f"[flex_checkpoint][fast_check] checkpoint '{path}' contains "
            "sharded tensors, the fast local resume check does not apply."
        )
        return None

    rank = paddle.distributed.get_rank() if use_dist else 0
    # Mirrors local_load_state_dict, which hardcodes unique_id 0. The verdict has
    # to predicate on the file the loader will actually read, so this name must
    # stay in lockstep with that one.
    ckpt_file = os.path.join(path, f"{rank}_0.distcp")
    if not os.path.isfile(ckpt_file):
        logger.info(
            f"[flex_checkpoint][fast_check] '{ckpt_file}' does not exist, this "
            "rank cannot resume locally."
        )
        local_load = False
    else:
        local_load = _file_matches_state_dict(ckpt_file, state_dict)

    if use_dist:
        global_local_loads = []
        paddle.distributed.all_gather_object(
            global_local_loads, local_load, process_group
        )
        verdict = all(global_local_loads)
        logger.info(
            f"[flex_checkpoint][fast_check] verdict={verdict} "
            f"(this rank={local_load}, "
            f"{sum(global_local_loads)}/{len(global_local_loads)} ranks agree) "
            f"decided in {time.time() - start:.3f}s without a full metadata "
            "load."
        )
        return verdict
    logger.info(
        f"[flex_checkpoint][fast_check] verdict={local_load} decided in "
        f"{time.time() - start:.3f}s without a full metadata load."
    )
    return local_load
