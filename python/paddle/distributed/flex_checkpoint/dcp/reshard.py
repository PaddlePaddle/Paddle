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

from collections import defaultdict
from typing import TYPE_CHECKING

import paddle.distributed as dist

from .load_state_dict import _load_state_dict
from .metadata import Metadata
from .utils import extract_sharded_weight_desc_and_tensor

if TYPE_CHECKING:
    from paddle.distributed.communication.group import Group

    from .sharded_weight import (
        ShardedWeight,
        ShardedWeightDesc,
    )


def reshard_sharded_weights(
    src_sharded_weights: list[ShardedWeight],
    dst_sharded_weights: list[ShardedWeight],
    process_group: Group,
):
    src_sharded_state_dict = {}
    dst_sharded_state_dict = {}
    for sharded_weight in src_sharded_weights:
        sharded_weight_desc, local_tensor = (
            extract_sharded_weight_desc_and_tensor(
                sharded_weight.key,
                sharded_weight,
            )
        )
        src_sharded_state_dict[sharded_weight_desc] = local_tensor

    for sharded_weight in dst_sharded_state_dict:
        sharded_weight_desc, local_tensor = (
            extract_sharded_weight_desc_and_tensor(
                sharded_weight.key,
                sharded_weight,
            )
        )
        dst_sharded_state_dict[sharded_weight_desc] = local_tensor

    # build metadata
    state_dict_metadata = defaultdict(list)
    for sharded_weight_desc, local_tensor in src_sharded_state_dict.items():
        state_dict_metadata[sharded_weight_desc.key].append(sharded_weight_desc)

    virtual_file_path = f"vfile_{dist.get_rank()}"
    local_storage_metadata = {
        sharded_weight_desc: virtual_file_path
        for sharded_weight_desc, local_tensor in src_sharded_state_dict.items()
    }

    global_storage_metadata: list[dict[ShardedWeightDesc, str]] = []
    dist.all_gather_object(
        global_storage_metadata,
        local_storage_metadata,
        group=process_group,
    )

    # Merge storage metadata
    storage_metadata: dict[ShardedWeightDesc, str] = {}
    for rank_storage_metadata in global_storage_metadata:
        storage_metadata.update(rank_storage_metadata)

    # Prepare metadata for loading
    metadata = Metadata(
        state_dict_metadata=state_dict_metadata,
        storage_metadata=storage_metadata,
        flat_mapping=None,
    )

    # reshard using _load_state_dict
    _load_state_dict(
        target_state_dict=dst_sharded_state_dict,
        source_state_dict={virtual_file_path: src_sharded_state_dict},
        metadata_list=[metadata],
        process_group=process_group,
    )
