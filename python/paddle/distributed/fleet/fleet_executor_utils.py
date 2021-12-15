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

from paddle.distributed.fleet.meta_optimizers.common import OpRole, OP_ROLE_KEY
from paddle.fluid import core


class CoordSys:
    """
    This class is used to mapping rank to (mp rank, sharding rank, pp rank, dp rank).
    """

    def __init__(self, dist_opt):
        self.dp_degree = dist_opt.get('dp_degree', 1)
        self.pp_degree = dist_opt.get('pp_degree', 1)
        self.sharding_degree = dist_opt.get('sharding_degree', 1)
        self.mp_degree = dist_opt.get('mp_degree', 1)

    def _invalide_coord(self, coord):
        """
        Test the input coord is valid or not.
        :param coord: The coord to be tested
        :return: False if valid, True if invalid.
        """
        return coord['mp_idx'] < 0 or coord['mp_idx'] >= self.mp_degree or \
               coord['sharding_idx'] < 0 or coord['sharding_idx'] >= self.sharding_degree or \
               coord['pp_idx'] < 0 or coord['pp_idx'] >= self.pp_degree or \
               coord['dp_idx'] < 0 or coord['dp_idx'] >= self.dp_degree

    def coord_to_rank(self, coord):
        """
        Map the input coord to it's corresponding rank.
        :param coord:  The coord to be converted
        :return: The rank corresponding with the coord
        """
        if self._invalide_coord(coord):
            return -1
        return int(coord['dp_idx'] * self.pp_degree * self.sharding_degree * self.mp_degree + \
                   coord['pp_idx'] * self.sharding_degree * self.mp_degree + \
                   coord['sharding_idx'] * self.mp_degree + coord['mp_idx'])

    def rank_to_coord(self, rank):
        """
        Map the input rank to it's corresponding coord
        :param rank: The rank to be converted
        :return: The coord corresponding with the rank
        """
        mp_idx = rank % self.mp_degree
        rank //= self.mp_degree
        sharding_idx = rank % self.sharding_degree
        rank //= self.sharding_degree
        pp_idx = rank % self.pp_degree
        rank //= self.pp_degree
        dp_idx = rank % self.dp_degree
        return {
            'mp_idx': int(mp_idx),
            'sharding_idx': int(sharding_idx),
            'pp_idx': int(pp_idx),
            'dp_idx': int(dp_idx)
        }


def is_optimizer_op(op_role):
    return op_role == int(OpRole.Optimize)


def is_lr_sched_op(op_role):
    return op_role == int(OpRole.Optimize.LRSched)


def is_forward_op(op_role):
    return (op_role == int(OpRole.Forward)) or \
           (op_role == (int(OpRole.Forward) ^ int(OpRole.Loss)))


def is_backward_op(op_role):
    return (op_role == int(OpRole.Backward)) or \
           (op_role == (int(OpRole.Backward) ^ int(OpRole.Loss)))


def one_f_one_b(program, cur_rank, max_run_times, dist_opt, nrank):
    """
    Split the program to support 1f1b pipeline scheduler.
    This funct will split the program based on the op_role.
    The program will be split into four parts: lr_sched, fwd, bwd, opt.
    And will create task nodes based on the four parts of the program.
    :param program: The origin program.
    :param cur_rank: Current rank (can be got from fleet.worker_index()).
    :param max_run_times: Max run times for a micro batch. AKA number of micro steps.
    :param dist_opt: The fleet_opt configured by user.
    :param nrank: Number of workers (can be got from fleet.worker_num()).
    :return:
        task_nodes (list): four task nodes for current rank
        task_id_to_rank (dict): task nodes' ids to it's corresponding rank
    """
    coord_sys = CoordSys(dist_opt)
    coord = coord_sys.rank_to_coord(cur_rank)
    max_slot_times = int(max_run_times - coord['pp_idx'])
    num_of_functionality = 4

    def create_task_node(role, ops, offset, node_type):
        task_id = int(cur_rank * num_of_functionality + offset)
        print("Creating task node with role: ", role, ", and with id: ",
              task_id)
        node = core.TaskNode(role, ops, cur_rank, task_id, max_run_times,
                             max_slot_times)
        node.set_type(node_type)
        return node

    lr_ops, fwd_ops, bwd_ops, opt_ops = [], [], [], []
    for op in program.block(0).ops:
        # split the program based on the op_role
        op_role = int(op.all_attrs()[OP_ROLE_KEY])
        if is_lr_sched_op(op_role):
            lr_ops.append(op.desc)
        elif is_optimizer_op(op_role):
            opt_ops.append(op.desc)
        elif is_forward_op(op_role):
            fwd_ops.append(op.desc)
        elif is_backward_op(op_role):
            bwd_ops.append(op.desc)
        else:
            raise "The op role: " + str(
                op_role
            ) + " isn't one of LRSched, Forward, Backward or Optimizer."

    # Create task nodes.
    # The lr_sched and opt should be 'amplifier interceptor.
    # The fwd and bwd should be 'compute interceptor'.
    lr_task_node = create_task_node(
        int(OpRole.Optimize.LRSched), lr_ops, 0, "Amplifier")
    lr_task_node.set_run_pre_steps(max_run_times)
    fwd_task_node = create_task_node(int(OpRole.Forward), fwd_ops, 1, "Compute")
    bwd_task_node = create_task_node(
        int(OpRole.Backward), bwd_ops, 2, "Compute")
    opt_task_node = create_task_node(
        int(OpRole.Optimize), opt_ops, 3, "Amplifier")
    opt_task_node.set_run_pre_steps(max_run_times)
    opt_task_node.set_run_at_offset(max_run_times - 1)
    task_nodes = [lr_task_node, fwd_task_node, bwd_task_node, opt_task_node]

    # Generated the dependency based on this graph:
    # lr(1:m) -> forward -> backward -> (m:1)optimize
    #               ↑          ↓
    # lr(1:m) -> forward -> backward -> (m:1)optimize
    #               ↑          ↓
    # lr(1:m) -> forward -> backward -> (m:1)optimize
    upstream_coord, downstream_coord = coord.copy(), coord.copy()
    upstream_coord['pp_idx'] = upstream_coord['pp_idx'] - 1
    downstream_coord['pp_idx'] = downstream_coord['pp_idx'] + 1
    pp_upstream = coord_sys.coord_to_rank(upstream_coord)
    pp_downstream = coord_sys.coord_to_rank(downstream_coord)
    first_stage = (pp_upstream == -1)
    last_stage = (pp_downstream == -1)
    for i in range(num_of_functionality):
        task_node = task_nodes[i]
        task_role = task_node.role()
        cur_id = int(cur_rank * num_of_functionality + i)
        prev_id = cur_id - 1
        next_id = cur_id + 1
        upstream_id = int(pp_upstream * num_of_functionality + i)
        downstream_id = int(pp_downstream * num_of_functionality + i)
        pp_buff_size = int(dist_opt['pp_degree'] - coord['pp_idx'])
        ups = []
        downs = []
        if not is_lr_sched_op(task_role):
            buf_size = pp_buff_size if is_backward_op(task_role) else 2
            ups.append((prev_id, buf_size))
        if not is_optimizer_op(task_role):
            buf_size = pp_buff_size if is_forward_op(task_role) else 2
            downs.append((next_id, buf_size))
        if is_forward_op(task_role):
            if not first_stage:
                ups.append((upstream_id, 2))
            if not last_stage:
                downs.append((downstream_id, 2))
        elif is_backward_op(task_role):
            if not last_stage:
                ups.append((downstream_id, 2))
            if not first_stage:
                downs.append((upstream_id, 2))
        for up in ups:
            print("Task: ", cur_id, "'s upstream includes: ", up[0], ".")
            task_node.add_upstream_task(up[0], up[1])
        for down in downs:
            print("Task: ", cur_id, "'s downstream includes: ", down[0], ".")
            task_node.add_downstream_task(down[0], down[1])
    task_id_to_rank = {}
    for i in range(nrank):
        for j in range(num_of_functionality):
            task_id_to_rank[int(i * num_of_functionality + j)] = i
    return task_nodes, task_id_to_rank
