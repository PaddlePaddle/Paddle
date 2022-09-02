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
from paddle.static import Program


class TaskNode:
    """
    Python side TaskNode, connection to the c++ side TaskNode
    """

    def __init__(self,
                 rank,
                 max_run_times,
                 max_slot_times,
                 role=None,
                 node_type=None,
                 task_id=0,
                 ops=None,
                 program=None,
                 lazy_initialize=False):
        """
        :param rank (int): Current rank of the task node.
        :param max_run_times (int): The max run times of the task node.
        :param max_slot_times (int): The mas slot times of the task node.
        :param role (int): The role of the task node. (Will be removed in the future)
        :param node_type (str): The type of the task node.
        :param task_id (int): The id of task node.
        :param ops (list): A list of op.desc to init the task node. (Will be removed in the future) 
        :param program (Program): An instance of Program to init the task node.
        :param lazy_initialize (bool): In user-defined task, the program may change adding feed/fetch op. As efficient consideration, the task node will have the C++ object later.
        """
        assert ((ops is not None) ^ (program is not None)), \
            "Should provide only one of ops or program to task node."
        assert (not ((ops is not None) and lazy_initialize)), \
                "Lazy initialization doesn't support with ops list"
        self.id = int(task_id)
        self.rank = rank
        self.max_run_times = max_run_times
        self.max_slot_times = max_slot_times
        self.node_type = node_type
        self.program = program
        self.lazy_initialize = lazy_initialize
        self.run_pre_steps = None
        self.run_at_offset = None
        self.node = None
        self.upstreams = []
        self.downstreams = []
        if not lazy_initialize:
            if ops is not None:
                assert role is not None and task_id is not None, \
                    "If init task node with ops, should provide `role` and `task_id`."
                self.node = core.TaskNode(role, ops, rank, task_id,
                                          max_run_times, max_slot_times)
            else:
                self.node = core.TaskNode(program.desc, rank, self.id,
                                          max_run_times, max_slot_times)
            if self.node_type:
                self.node.set_type(self.node_type)

    def task_node(self):
        if self.lazy_initialize:
            self.node = core.TaskNode(self.program.desc, self.rank, self.id,
                                      self.max_run_times, self.max_slot_times)
            if self.node_type:
                self.node.set_type(self.node_type)
            if self.run_pre_steps:
                self.node.set_run_pre_steps(self.run_pre_steps)
            if self.run_at_offset:
                self.node.set_run_at_offset(self.run_at_offset)
            for up in self.upstreams:
                self.node.add_upstream_task(up[0], up[1])
            for down in self.downstreams:
                self.node.add_downstream_task(down[0], down[1])
            self.lazy_initialize = False
        return self.node

    def set_program(self, program):
        assert self.lazy_initialize, \
            "Inside program is unchangable for immediate initialized task node. Set the lazy_initialize to be true if the inside program need to be update. Remember to do all your change before eval node.task_node()."
        self.program = program

    def get_program(self):
        assert self.program is not None, "The task node is not initialized using program"
        return self.program

    def set_run_pre_steps(self, steps):
        if self.lazy_initialize:
            self.run_pre_steps = steps
        else:
            self.node.set_run_pre_steps(steps)

    def set_run_at_offset(self, offset):
        if self.lazy_initialize:
            self.run_at_offset = offset
        else:
            self.node.set_run_at_offset(offset)

    def add_upstream_task(self, upstream, buffer_size=2):
        if self.lazy_initialize:
            self.upstreams.append((upstream, buffer_size))
        else:
            self.node.add_upstream_task(upstream, buffer_size)

    def add_downstream_task(self, downstream, buffer_size=2):
        if self.lazy_initialize:
            self.downstreams.append((downstream, buffer_size))
        else:
            self.node.add_downstream_task(downstream, buffer_size)

    def task_id(self):
        return self.id


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


class FleetExecutorUtils:

    def __init__(self,
                 dist_strategy=None,
                 rank=None,
                 nrank=None,
                 max_run_times=None):
        self.dist_strategy = dist_strategy
        self.rank = rank
        self.nrank = nrank
        self.max_run_times = max_run_times
        self.is_auto_parallel = True if dist_strategy is None else False
        self.num_of_functionality = 4
        self.coord_sys = None
        self.coord = None
        if dist_strategy:
            self.coord_sys = CoordSys(dist_strategy)
            self.coord = self.coord_sys.rank_to_coord(rank)

    def is_optimizer_op(self, op_role):
        return op_role == int(OpRole.Optimize)

    def is_lr_sched_op(self, op_role):
        return op_role == int(OpRole.Optimize.LRSched)

    def is_forward_op(self, op_role):
        return (op_role == int(OpRole.Forward)) or \
               (op_role == (int(OpRole.Forward) | int(OpRole.Loss)))

    def is_backward_op(self, op_role):
        return (op_role == int(OpRole.Backward)) or \
               (op_role == (int(OpRole.Backward) | int(OpRole.Loss)))

    def split_program_to_op_list(self, program):
        op_list_map = {"lr": [], "fwd": [], "bwd": [], "opt": []}
        for op in program.block(0).ops:
            # split the program based on the op_role
            op_role = int(op.all_attrs()[OP_ROLE_KEY])
            if self.is_lr_sched_op(op_role):
                op_list_map["lr"].append(op)
            elif self.is_forward_op(op_role):
                op_list_map["fwd"].append(op)
            elif self.is_backward_op(op_role):
                op_list_map["bwd"].append(op)
            elif self.is_optimizer_op(op_role):
                op_list_map["opt"].append(op)
            else:
                raise "The op role: " + str(
                    op_role
                ) + " isn't one of LRSched, Forward, Backward or Optimizer."
        return op_list_map

    def convert_op_list_to_program(self, op_list, complete_program):
        #TODO(liyurui): Complete this convert logic
        program_map = {
            "lr": Program(),
            "fwd": Program(),
            "bwd": Program(),
            "opt": Program()
        }
        return program_map

    def build_1f1b_dependency(self, task_node_map):
        assert not self.is_auto_parallel, "Handly add dependency should not be invoked in auto parallel mode"
        # Generated the dependency based on this graph:
        # lr(1:m) -> forward -> backward -> (m:1)optimize
        #               ↑          ↓
        # lr(1:m) -> forward -> backward -> (m:1)optimize
        #               ↑          ↓
        # lr(1:m) -> forward -> backward -> (m:1)optimize

        # add dependency intra stage
        cur_start_id = self.rank * self.num_of_functionality
        pp_buff_size = int(self.dist_strategy['pp_degree'] -
                           self.coord['pp_idx'])
        task_node_map["lr"].add_downstream_task(cur_start_id + 1)
        task_node_map["fwd"].add_upstream_task(cur_start_id)
        task_node_map["fwd"].add_downstream_task(cur_start_id + 2, pp_buff_size)
        task_node_map["bwd"].add_upstream_task(cur_start_id + 1, pp_buff_size)
        task_node_map["bwd"].add_downstream_task(cur_start_id + 3)
        task_node_map["opt"].add_upstream_task(cur_start_id + 2)
        # add dependency inter stage
        upstream_coord, downstream_coord = self.coord.copy(), self.coord.copy()
        upstream_coord['pp_idx'] = upstream_coord['pp_idx'] - 1
        downstream_coord['pp_idx'] = downstream_coord['pp_idx'] + 1
        pp_upstream = self.coord_sys.coord_to_rank(upstream_coord)
        pp_downstream = self.coord_sys.coord_to_rank(downstream_coord)
        first_stage = (pp_upstream == -1)
        last_stage = (pp_downstream == -1)
        prev_pp_start_id = pp_upstream * self.num_of_functionality
        next_pp_start_id = pp_downstream * self.num_of_functionality
        if not first_stage:
            task_node_map["fwd"].add_upstream_task(prev_pp_start_id + 1)
            task_node_map["bwd"].add_downstream_task(prev_pp_start_id + 2)
        if not last_stage:
            task_node_map["fwd"].add_downstream_task(next_pp_start_id + 1)
            task_node_map["bwd"].add_upstream_task(next_pp_start_id + 2)
        return task_node_map

    def construct_task_nodes_1f1b(self, program_map):
        max_slot_times = int(self.max_run_times - self.coord['pp_idx'])
        cur_start_id = int(self.rank * self.num_of_functionality)
        lr_task_node = TaskNode(rank=self.rank,
                                max_run_times=self.max_run_times,
                                max_slot_times=max_slot_times,
                                program=program_map["lr"],
                                task_id=cur_start_id)
        fwd_task_node = TaskNode(rank=self.rank,
                                 max_run_times=self.max_run_times,
                                 max_slot_times=max_slot_times,
                                 program=program_map["fwd"],
                                 task_id=cur_start_id + 1)
        bwd_task_node = TaskNode(rank=self.rank,
                                 max_run_times=self.max_run_times,
                                 max_slot_times=max_slot_times,
                                 program=program_map["bwd"],
                                 task_id=cur_start_id + 2)
        opt_task_node = TaskNode(rank=self.rank,
                                 max_run_times=self.max_run_times,
                                 max_slot_times=max_slot_times,
                                 program=program_map["opt"],
                                 task_id=cur_start_id + 3)
        return {
            "lr": lr_task_node,
            "fwd": fwd_task_node,
            "bwd": bwd_task_node,
            "opt": opt_task_node
        }

    def task_id_to_rank(self):
        task_id_to_rank = {}
        for i in range(self.nrank):
            for j in range(self.num_of_functionality):
                task_id_to_rank[int(i * self.num_of_functionality + j)] = i
        return task_id_to_rank

    def construct_task_nodes_1f1b_op_list(self, op_list_map):
        max_slot_times = int(self.max_run_times - self.coord['pp_idx'])
        cur_start_id = int(self.rank * self.num_of_functionality)
        lr_task_node = TaskNode(rank=self.rank,
                                max_run_times=self.max_run_times,
                                max_slot_times=max_slot_times,
                                role=int(OpRole.Optimize.LRSched),
                                ops=op_list_map["lr"],
                                task_id=cur_start_id,
                                node_type="Amplifier")
        lr_task_node.set_run_pre_steps(self.max_run_times)
        fwd_task_node = TaskNode(rank=self.rank,
                                 max_run_times=self.max_run_times,
                                 max_slot_times=max_slot_times,
                                 role=int(OpRole.Forward),
                                 ops=op_list_map["fwd"],
                                 task_id=cur_start_id + 1,
                                 node_type="Compute")
        bwd_task_node = TaskNode(rank=self.rank,
                                 max_run_times=self.max_run_times,
                                 max_slot_times=max_slot_times,
                                 role=int(OpRole.Backward),
                                 ops=op_list_map["bwd"],
                                 task_id=cur_start_id + 2,
                                 node_type="Compute")
        opt_task_node = TaskNode(rank=self.rank,
                                 max_run_times=self.max_run_times,
                                 max_slot_times=max_slot_times,
                                 role=int(OpRole.Optimize),
                                 ops=op_list_map["opt"],
                                 task_id=cur_start_id + 3,
                                 node_type="Amplifier")
        opt_task_node.set_run_pre_steps(self.max_run_times)
        opt_task_node.set_run_at_offset(self.max_run_times - 1)
        return {
            "lr": lr_task_node,
            "fwd": fwd_task_node,
            "bwd": bwd_task_node,
            "opt": opt_task_node
        }


def run1f1b(program,
            rank,
            max_run_times,
            dist_opt,
            nrank,
            with_standalone_executor=False):
    """
    Split the program to support 1f1b pipeline scheduler.
    This funct will split the program based on the op_role.
    The program will be split into four parts: lr_sched, fwd, bwd, opt.
    And will create task nodes based on the four parts of the program.
    :param program: The origin program.
    :param rank: Current rank (can be got from fleet.worker_index()).
    :param max_run_times: Max run times for a micro batch. AKA number of micro steps.
    :param dist_opt: The fleet_opt configured by user.
    :param nrank: Number of workers (can be got from fleet.worker_num()).
    :param with_standalone_executor: Experiment feature, use fleet executor with standalone executor.
    :return:
        task_nodes (list): four task nodes for current rank
        task_id_to_rank (dict): task nodes' ids to it's corresponding rank
    """
    print("fleet executor will use python side 1f1b scheduler.")
    fleet_executor_utils = FleetExecutorUtils(dist_strategy=dist_opt,
                                              rank=rank,
                                              nrank=nrank,
                                              max_run_times=max_run_times)
    op_list_map = fleet_executor_utils.split_program_to_op_list(program)
    task_node_map = None
    if with_standalone_executor:
        program_map = fleet_executor_utils.convert_op_list_to_program(
            op_list_map, program)
        task_node_map = fleet_executor_utils.construct_task_nodes_1f1b(
            program_map)
    else:
        op_desc_list_map = {"lr": [], "fwd": [], "bwd": [], "opt": []}
        for key in op_list_map:
            for op in op_list_map[key]:
                op_desc_list_map[key].append(op.desc)
        task_node_map = fleet_executor_utils.construct_task_nodes_1f1b_op_list(
            op_desc_list_map)
    task_node_map = fleet_executor_utils.build_1f1b_dependency(task_node_map)
    task_id_to_rank = fleet_executor_utils.task_id_to_rank()
    task_node_list = [task_node_map[key].task_node() for key in task_node_map]
    return task_node_list, task_id_to_rank


def origin(program, rank):
    """
    Origin scheduler for fleet executor, supports non-pp mode
    :param program: The origin program.
    :param rank: Current rank (can be got from fleet.worker_index()).
    :return:
        task_nodes (list): four task nodes for current rank
        task_id_to_rank (dict): a fake dict, since there is no upstream or downstream, this dict won't be used
    """
    print("fleet executor will use python side origin scheduler.")
    task_node = TaskNode(program=program,
                         rank=rank,
                         node_type="Compute",
                         max_run_times=1,
                         max_slot_times=1)
    task_id_to_rank = {task_node.task_id(): rank}
    return [task_node.task_node()], task_id_to_rank
