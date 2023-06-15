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
import logging

import paddle

from ..utils.log_utils import get_logger
from .process_mesh import retrive_unique_id_for_process_mesh
from .static.utils import _get_idx_in_axis

_logger = get_logger(logging.INFO)

_rng_name_to_seed = {}
_inited_rng_name_to_seed = {}
_enable_random_control = False
_basic_seed = 42

# use Prime number as offset to avoid confict
_mesh_offset = 173
_dim_offsets = [11, 23, 37, 73]


def is_enable_auto_rand_ctrl():
    global _enable_random_control
    return _enable_random_control


def enable_auto_rand_ctrl():
    global _enable_random_control
    _enable_random_control = True


def parallel_manual_seed(seed):
    """Enable auto parallel random control.
    Random control maintain the randomness when tensor is distributed across devices on a Mesh(any order).
        * Independency: If tensor is **Sharded** on a Mesh dimension, Devices along that Mesh dimension should have Different randomness.

        * Consistency:  Meanwhile if the tensor is **Replicated** on another Mesh dimension, randomness of Devices along that Mesh dimension should be Consistent.

    For instance: rank0 ~ rank7 consist a Mesh of shape of [2, 4]; A 2D tensor is distributed in that Mesh using dims_mapping [-1, 1].
    Randomness for rank0-rank1-rank2-rank3 (rank4-rank5-rank6-rank7) should be Independent;
    Randomness for rank0 and rank4 (rank1 and rank5, ...) should be Consistent.

    This function should be called only once before auto parallel compiles the computation graph (e.g. auto_parallel.engine.prepare() or fit()).

    This seed only affects how randomness-relative **operators** (dropout, fuse op with dropout inside, etc) are execute amonge mesh, and would NOT affect other processe like Parameter initialization.

    Examples:
        # seed relative to training step
        auto_parallel_random_seed((step + 13) * 257)
        ...
        engine.prepare()
    """

    enable_auto_rand_ctrl()
    global _basic_seed
    _basic_seed = seed


def determinate_rng(rank, dims_mapping, process_mesh):

    # TODO(JZ-LIANG) Support Mesh with any high rank
    # use a string to unique integer hashing algorithm for seed computation.
    # instead of using offsets to coodinate seed across devices.
    if len(process_mesh.shape) > 4:
        raise NotImplementedError(
            "Auto Parallel Random Control for Mesh's rank > 4 is NOT supported! Got {}".format(
                str(process_mesh)
            )
        )
    global _basic_seed
    seed_ = _basic_seed

    # FIXME
    # unique_id = process_mesh.unique_id
    unique_id = retrive_unique_id_for_process_mesh(
        process_mesh.shape, process_mesh.process_ids
    )
    sharding_expr = f'mesh:{unique_id}'
    seed_ += _mesh_offset * (unique_id + 1)

    for i in range(len(process_mesh.shape)):
        if i not in dims_mapping:
            relative_idx = -1
        else:
            relative_idx = _get_idx_in_axis(
                process_mesh.process_ids,
                process_mesh.shape,
                i,
                rank,
            )

        sharding_expr += f"_dim{i}:{relative_idx}"
        seed_ += _dim_offsets[i] * (relative_idx + 1)

    global _rng_name_to_seed
    if sharding_expr in _rng_name_to_seed:
        assert _rng_name_to_seed[sharding_expr] == seed_
    else:
        assert (
            seed_ not in _rng_name_to_seed.values()
        ), "Seed Confilt! current seed: {}, current sharding expr: {}, generated seed: {}".format(
            seed_, sharding_expr, _rng_name_to_seed
        )
        _rng_name_to_seed[sharding_expr] = seed_

    return sharding_expr


def init_auto_parallel_rng():

    if not is_enable_auto_rand_ctrl():
        return

    global _rng_name_to_seed
    # NOTE init rng maybe call multiple times, avoid init same rng twice
    global _inited_rng_name_to_seed

    for rng_name, seed in _rng_name_to_seed.items():
        if rng_name in _inited_rng_name_to_seed:
            assert _inited_rng_name_to_seed[rng_name] == seed
        else:
            _logger.info(
                f"Init Auto Parallel RNG: {rng_name}, with seed {seed}"
            )
            paddle.framework.random.set_random_seed_generator(rng_name, seed)
            _inited_rng_name_to_seed[rng_name] = seed
