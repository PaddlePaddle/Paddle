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

from .meta_optimizers.common import OpRole, OP_ROLE_KEY
from paddle.fluid import core


class CoordSys:
    def __init__(self, dist_opt):
        self.dp_degree = dist_opt.get('dp_degree', 1)
        self.pp_degree = dist_opt.get('pp_degree', 1)
        self.sharding_degree = dist_opt.get('sharding_degree', 1)
        self.mp_degree = dist_opt.get('mp_degree', 1)

    def _invalide_coord(self, coord):
        return coord['mp_idx'] < 0 or coord['mp_idx'] >= self.mp_degree or \
               coord['sharding_idx'] < 0 or coord['sharding_idx'] >= self.sharding_degree or \
               coord['pp_idx'] < 0 or coord['pp_idx'] >= self.pp_degree or \
               coord['dp_idx'] < 0 or coord['dp_idx'] >= self.dp_degree

    def coord_to_rank(self, coord):
        if self._invalide_coord(coord):
            return -1
        return coord['dp_idx'] * self.pp_degree * self.sharding_degree * self.mp_degree + \
               coord['pp_idx'] * self.sharding_degree * self.mp_degree + \
               coord['sharding_idx'] * self.mp_degree + coord['mp_idx']

    def rank_to_coord(self, rank):
        mp_idx = rank % self.mp_degree
        rank /= self.mp_degree
        sharding_idx = rank % self.sharding_degree
        rank /= self.sharding_degree
        pp_idx = rank % self.pp_degree
        rank /= self.pp_degree
        dp_idx = rank % self.dp_degree
        return {
            'mp_idx': mp_idx,
            'sharding_idx': sharding_idx,
            'pp_idx': pp_idx,
            'dp_idx': dp_idx
        }


def one_f_one_b(program, cur_rank, max_run_times, dist_opt):
    coord_sys = CoordSys(dist_opt)
    coord = coord_sys.rank_to_coord(cur_rank)
    max_slot_times = max_run_times - coord['pp_idx']
    lr_ops, fwd_ops, bwd_ops, opt_ops = [], [], [], []
    num_of_functionality = 4
    for op in program.block(0).ops:
        op_role = int(op.all_attrs()[OP_ROLE_KEY])
        if op_role == int(OpRole.Optimize.LRSched):
            lr_ops.append(op)
        elif op_role == int(OpRole.Optimize):
            opt_ops.append(op)
        elif op_role == int(OpRole.Forward) or op_role == (int(OpRole.Forward) ^
                                                           int(OpRole.Loss)):
            fwd_ops.append(op)
        elif op_role == int(OpRole.Backward) or op_role == (int(OpRole.Backward)
                                                            ^ int(OpRole.Loss)):
            bwd_ops.append(op)
        else:
            raise "The op role: " + str(
                op_role
            ) + " isn't one of LRSched, forward, backward or optimizer."
    lr_task_node = core.TaskNode(
        int(OpRole.Optimize.LRSched), lr_ops, cur_rank,
        cur_rank * num_of_functionality, max_run_times, max_slot_times)
    lr_task_node.set_type("Amplifier")
    lr_task_node.set_run_pre_steps(max_run_times)
    fwd_task_node = core.TaskNode(
        int(OpRole.Forward), fwd_ops, cur_rank,
        cur_rank * num_of_functionality + 1, max_run_times, max_slot_times)
    fwd_task_node.set_type("Compute")
    bwd_task_node = core.TaskNode(
        int(OpRole.Backward), bwd_ops, cur_rank,
        cur_rank * num_of_functionality + 2, max_run_times, max_slot_times)
    bwd_task_node.set_type("Compute")
    opt_task_node = core.TaskNode(
        int(OpRole.Optimize), opt_ops, cur_rank,
        cur_rank * num_of_functionality + 3, max_run_times, max_slot_times)
    opt_task_node.set_type("Amplifier")
    opt_task_node.set_run_pre_steps(max_run_times)
    opt_task_node.set_run_at_offset(max_run_times - 1)

    # lr(1:m) -> forward -> backward -> (m:1)optimize
    #               ↑          ↓
    # lr(1:m) -> forward -> backward -> (m:1)optimize
    #               ↑          ↓
    # lr(1:m) -> forward -> backward -> (m:1)optimize
    upstream_coord, downstream_coord = coord, coord
    upstream_coord['pp_idx'] -= 1
    downstream_coord['pp_idx'] += 1
    pp_upstream = coord_sys.coord_to_rank(upstream_coord)
    pp_downstream = coord_sys.coord_to_rank(downstream_coord)
    first_stage = (pp_upstream == -1)
    last_stage = (pp_downstream == -1)
    tmp = [lr_task_node, fwd_task_node, bwd_task_node, opt_task_node]
    for i in range(4):
        cur_task_node = tmp[i]
        cur_id = cur_rank * num_of_functionality + i
        prev_id = cur_id - 1
        next_id = cur_id + 1
        upstream_id = pp_upstream * num_of_functionality + i
        downstream_id = pp_downstream * num_of_functionality + i
        pp_buff_size = dist_opt['pp_degree'] - coord['pp_idx']
        ups = []
        downs = []
        if i != 0:
            buf_size = pp_buff_size if i == 2 else 2
            ups.append((prev_id, buf_size))
        if i != 3:
            buf_size = pp_buff_size if i == 1 else 2
            downs.append((next_id, buf_size))
        if i == 1:
            if not first_stage:
                ups.append((upstream_id, 2))
            if not last_stage:
                downs.append((downstream_id, 2))
        elif i == 2:
            if not last_stage:
                ups.append((downstream_id, 2))
            if not first_stage:
                downs.append((upstream_id, 2))
        for up in ups:
            cur_task_node.add_upstream_task(up[0], up[1])
        for down in downs:
            cur_task_node.add_downstream_task(down[0], down[1])
    return tmp
