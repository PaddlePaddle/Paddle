# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

_groups = None
_hcg = None


def initialize_p2p_groups(hcg):
    global _groups, _hcg
    _groups = [
        paddle.distributed.new_group(ranks=group)
        for group in hcg.get_p2p_groups()
    ]
    _hcg = hcg


def _is_valid_communciate(src_stage, dest_stage):
    first_stage = 0
    last_stage = _hcg.get_pipe_parallel_world_size() - 1
    assert abs(src_stage-dest_stage) == 1 or \
        (src_stage == first_stage and dest_stage == last_stage) or \
        (src_stage == last_stage and dest_stage == first_stage)


def send(tensor, dest_stage):
    global _groups, _hcg
    src_stage = _hcg.get_stage_id()
    src_rank = _hcg.get_rank_from_stage(stage_id=src_stage)

    _is_valid_communciate(src_stage, dest_stage)
    group = _get_send_recv_group(src_stage, dest_stage)
    dst_rank = _hcg.get_rank_from_stage(stage_id=dest_stage)
    return paddle.distributed.broadcast(tensor, src_rank, group=group)


def recv(tensor, src_stage):
    global _groups, _hcg
    dest_stage = _hcg.get_stage_id()

    _is_valid_communciate(src_stage, dest_stage)
    group = _get_send_recv_group(src_stage, dest_stage)
    src_rank = _hcg.get_rank_from_stage(stage_id=src_stage)
    return paddle.distributed.broadcast(tensor, src_rank, group=group)


def _is_valid_communciate(src_stage, dest_stage):
    first_stage = 0
    last_stage = _hcg.get_pipe_parallel_world_size() - 1
    assert abs(src_stage-dest_stage) == 1 or \
        (src_stage == first_stage and dest_stage == last_stage) or \
        (src_stage == last_stage and dest_stage == first_stage)


def _get_send_recv_group(src_stage, dest_stage):
    global _groups, _hcg
    stage_id = None
    first_stage = 0
    last_stage = _hcg.get_pipe_parallel_world_size() - 1
    if (src_stage == first_stage and dest_stage == last_stage) or \
            (dest_stage == first_stage and src_stage == last_stage):
        stage_id = last_stage
    elif src_stage > dest_stage:
        stage_id = dest_stage
    else:
        stage_id = src_stage
    group_id = _hcg.get_rank_from_stage(stage_id=stage_id)
    return _groups[group_id]
