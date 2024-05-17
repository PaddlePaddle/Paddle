#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License

import copy
from collections import defaultdict

from paddle.distributed.passes.pass_base import PassContext
from paddle.framework import IrGraph, core, set_flags

from ..process_mesh import ProcessMesh
from .dist_op import DistributedOperator
from .dist_tensor import DistributedTensor
from .utils import (
    __no_shape_var_type__,
    _copy_dist_attr_to_cpp,
    is_loss_grad_op,
)

# There always exists a default context for user. And user can set it to another one.
_g_default_distributed_context = None


def get_default_distributed_context():
    global _g_default_distributed_context
    if _g_default_distributed_context is None:
        dist_context = DistributedContext()
        set_default_distributed_context(dist_context)
    return _g_default_distributed_context


def set_default_distributed_context(dist_context):
    global _g_default_distributed_context
    _g_default_distributed_context = dist_context


def _node_id(node):
    return (node.node.graph_id(), node.node.id())


class DistributedContext:
    """
    DistributedContext is used to collect related distributed information for program and graph.
    One auto-parallel run should use its own DistributedContext to avoid interfering other run.
    """

    def __init__(
        self,
        serial_main_prog=None,
        serial_startup_prog=None,
        serial_optimizer=None,
        serial_loss=None,
        feed_vars={},
        fetch_vars={},
        cluster=None,
        strategy=None,
        json_config=None,
    ):
        # Data members related to original programs (unchanged)
        self._original_serial_main_program = serial_main_prog
        self._original_serial_startup_program = serial_startup_prog
        self._original_serial_optimizer = serial_optimizer
        self._original_serial_loss = serial_loss
        self._original_serial_feed_vars = feed_vars
        self._original_serial_fetch_vars = fetch_vars

        # Data members related to programs (changed)
        self._serial_main_program = None
        self._serial_startup_program = None
        self._serial_loss = None
        self._serial_optimizer = None
        self._serial_feed_vars = {}
        self._serial_fetch_vars = {}
        self._lr_optimizer = None  # record the optimizer holding lr_scheduler

        # Data members related to the program
        self._dist_tensors_for_program = {}
        self._dist_ops_for_program = {}

        # Data members related to the graph
        self._serial_graph = None
        self._dist_tensors_for_graph = {}
        self._dist_ops_for_graph = {}
        self._node_id_to_tensor_id = {}
        self._node_id_to_op_id = {}

        # Data members related to the distributed programs
        # Distributed programs
        self._dist_main_programs = {}
        self._dist_startup_programs = {}
        self._dist_op_context = DistributedOperatorContext()
        self._process_meshes = []

        self._cluster = cluster
        self._strategy = strategy

        # Pass Context
        self._pass_context = PassContext()
        self._block_state = BlockState()

        # Other data members
        self._serial_ordered_tensor_nodes = []
        self._serial_ordered_op_nodes = []
        self._serial_ordered_nodes = []
        # self._tensor_id_to_tensor_node_ids = {}

        self._is_initialized = False
        # TODO: need a better way to remove the following flag
        self._need_copy_dist_attr_to_graph = False
        self._backup_pass_context_stack = []
        self._backup_block_state_stack = []
        self._backup_dist_tensors_for_program_stack = []
        self._backup_dist_ops_for_program_stack = []
        self._backup_serial_main_program_stack = []
        self._backup_serial_startup_program_stack = []

        # flag whether scale gradient with dp size
        self._gradient_scale = True

        # whether use allreduce_avg to scale gradient, i.e., allreduce_sum + scale -> allreduce_avg
        self._gradient_scale_using_allreduce_avg = False

        # A flag indicates whether the used parallelism is data parallel
        self._data_parallel = False

        # record upstream and downstream of cur rank
        self._up_down_streams = UpDownStream()

        self._json_config = json_config

        # record vpp chunk size
        self._num_model_chunks = 0

    @property
    def serial_main_program(self):
        return self._serial_main_program

    @property
    def serial_startup_program(self):
        return self._serial_startup_program

    @property
    def serial_loss(self):
        return self._serial_loss

    @property
    def serial_optimizer(self):
        return self._serial_optimizer

    @property
    def serial_feed_vars(self):
        return self._serial_feed_vars

    @property
    def serial_fetch_vars(self):
        return self._serial_fetch_vars

    @property
    def dist_main_programs(self):
        return self._dist_main_programs

    @property
    def dist_startup_programs(self):
        return self._dist_startup_programs

    @property
    def cluster(self):
        return self._cluster

    @property
    def strategy(self):
        return self._strategy

    @property
    def serial_graph(self):
        return self._serial_graph

    @property
    def serial_ordered_nodes(self):
        return self._serial_ordered_nodes

    @property
    def process_meshes(self):
        return self._process_meshes

    @process_meshes.setter
    def process_meshes(self, val):
        self._process_meshes = val

    @property
    def pass_context(self):
        return self._pass_context

    @property
    def dist_op_context(self):
        return self._dist_op_context

    @property
    def block_state(self):
        return self._block_state

    @property
    def has_annotation(self):
        return len(self._dist_tensors_for_program) or len(
            self._dist_ops_for_program
        )

    @property
    def gradient_scale(self):
        return self._gradient_scale

    @gradient_scale.setter
    def gradient_scale(self, gs):
        self._gradient_scale = gs

    @property
    def gradient_scale_using_allreduce_avg(self):
        return self._gradient_scale_using_allreduce_avg

    @gradient_scale_using_allreduce_avg.setter
    def gradient_scale_using_allreduce_avg(
        self, gradient_scale_using_allreduce_avg
    ):
        self._gradient_scale_using_allreduce_avg = (
            gradient_scale_using_allreduce_avg
        )

    @property
    def data_parallel(self):
        return self._data_parallel

    @property
    def up_down_streams(self):
        return self._up_down_streams

    @data_parallel.setter
    def data_parallel(self, dp):
        self._data_parallel = dp

    def _backup_serial_info(self, mode):
        self._backup_serial_main_program_stack.append(
            self._serial_main_program.clone()
        )
        self._backup_serial_startup_program_stack.append(
            self._serial_startup_program.clone()
        )
        self._backup_pass_context_stack.append(
            copy.deepcopy(self._pass_context)
        )
        self._backup_block_state_stack.append(copy.deepcopy(self._block_state))

    def _backup_dist_info(self, mode):
        self._backup_dist_tensors_for_program_stack.append(
            copy.deepcopy(self._dist_tensors_for_program)
        )
        self._backup_dist_ops_for_program_stack.append(
            copy.deepcopy(self._dist_ops_for_program)
        )

    def _backup(self, serial=True, serial_mode=None, dist=True, dist_mode=None):
        # Use this function carefully
        if serial:
            self._backup_serial_info(serial_mode)
        if dist:
            self._backup_dist_info(dist_mode)

    def _restore_serial_loss(self):
        if self._original_serial_loss:
            if isinstance(self._original_serial_loss, list):
                if len(self._original_serial_loss) == 1:
                    loss = self._original_serial_loss[0]
                    block_idx = loss.block.idx
                    var_name = loss.name
                    var = self._serial_main_program.blocks[
                        block_idx
                    ]._var_recursive(var_name)
                    self._serial_loss = var
                elif len(self._original_serial_loss) == 0:
                    self._serial_loss = []
                else:
                    raise ValueError("multi loss vars are not supported.")
            else:
                block_idx = self._original_serial_loss.block.idx
                var_name = self._original_serial_loss.name
                var = self._serial_main_program.blocks[
                    block_idx
                ]._var_recursive(var_name)
                self._serial_loss = var

    def _restore_serial_feed_vars(self):
        for key, var_list in self._original_serial_feed_vars.items():
            new_var_list = []
            for var in var_list:
                block_idx = var.block.idx
                var_name = var.name
                var = self._serial_main_program.blocks[
                    block_idx
                ]._var_recursive(var_name)
                new_var_list.append(var)
            self._serial_feed_vars[key] = new_var_list

    def _restore_serial_fetch_vars(self):
        for key, var_list in self._original_serial_fetch_vars.items():
            new_var_list = []
            # metrics is a list of list
            if key == "metrics":
                for inner_var_list in var_list:
                    new_inner_var_list = []
                    for var in inner_var_list:
                        block_idx = var.block.idx
                        var_name = var.name
                        var = self._serial_main_program.blocks[
                            block_idx
                        ]._var_recursive(var_name)
                        new_inner_var_list.append(var)
                    new_var_list.append(new_inner_var_list)
            else:
                for var in var_list:
                    block_idx = var.block.idx
                    var_name = var.name
                    var = self._serial_main_program.blocks[
                        block_idx
                    ]._var_recursive(var_name)
                    new_var_list.append(var)
            self._serial_fetch_vars[key] = new_var_list

    def _restore_serial_info(self, mode="to_backup"):
        if mode == "to_backup":
            self._serial_main_program = (
                self._backup_serial_main_program_stack.pop()
            )
            self._serial_startup_program = (
                self._backup_serial_startup_program_stack.pop()
            )
        elif mode == "to_original":
            assert self._original_serial_main_program is not None
            assert self._original_serial_startup_program is not None
            self._serial_main_program = (
                self._original_serial_main_program.clone()
            )
            self._serial_startup_program = (
                self._original_serial_startup_program.clone()
            )

        self._restore_serial_loss()
        self._restore_serial_feed_vars()
        self._restore_serial_fetch_vars()
        self._serial_optimizer = self._original_serial_optimizer
        self._pass_context = self._backup_pass_context_stack.pop()
        self._block_state = self._backup_block_state_stack.pop()

    def _restore_dist_info(self, mode="to_backup"):
        if mode == "to_backup":
            self._dist_tensors_for_program = (
                self._backup_dist_tensors_for_program_stack.pop()
            )
            self._dist_ops_for_program = (
                self._backup_dist_ops_for_program_stack.pop()
            )
        elif mode == "to_original":
            assert self._original_dist_tensors_for_program
            assert self._original_dist_ops_for_program
            self._dist_tensors_for_program = copy.deepcopy(
                self._original_dist_tensors_for_program
            )
            self._dist_ops_for_program = copy.deepcopy(
                self._original_dist_ops_for_program
            )
        elif mode == "to_default":
            new_tensors_ids = []
            for (
                tensor_id,
                dist_tensor,
            ) in self._dist_tensors_for_program.items():
                if tensor_id in self._tensors_ids:
                    dist_tensor.dist_attr.reset()
                else:
                    new_tensors_ids.append(tensor_id)
            for tensor_id in new_tensors_ids:
                self._dist_tensors_for_program.pop(tensor_id)
            new_ops_ids = []
            for op_id, dist_op in self._dist_ops_for_program.items():
                if op_id in self._ops_ids:
                    dist_op.dist_attr.reset()
                else:
                    new_ops_ids.append(op_id)
            for op_id in new_ops_ids:
                self._dist_ops_for_program.pop(op_id)
        else:
            new_tensors_ids = []
            for (
                tensor_id,
                dist_tensor,
            ) in self._dist_tensors_for_program.items():
                new_tensors_ids.append(tensor_id)
            for tensor_id in new_tensors_ids:
                self._dist_tensors_for_program.pop(tensor_id)
            new_ops_ids = []
            for op_id, dist_op in self._dist_ops_for_program.items():
                new_ops_ids.append(op_id)
            for op_id in new_ops_ids:
                self._dist_ops_for_program.pop(op_id)
        self._dist_main_programs = {}
        self._dist_startup_programs = {}
        self._dist_op_context = DistributedOperatorContext()
        self._need_copy_dist_attr_to_graph = True
        self._process_meshes = []

    def _restore(
        self,
        serial=True,
        serial_mode="to_backup",
        dist=True,
        dist_mode="to_backup",
    ):
        # Use this function carefully
        if serial:
            self._restore_serial_info(serial_mode)
        if dist:
            self._restore_dist_info(dist_mode)

    def initialize(self, with_graph=True, with_cpp=False, no_default=False):
        if not self._is_initialized:
            if not self._serial_main_program:
                if self._original_serial_main_program:
                    self._serial_main_program = (
                        self._original_serial_main_program.clone()
                    )
            if not self._serial_startup_program:
                if self._original_serial_startup_program:
                    self._serial_startup_program = (
                        self._original_serial_startup_program.clone()
                    )
            if not self._serial_loss:
                self._restore_serial_loss()
            if not self._serial_optimizer:
                self._serial_optimizer = self._original_serial_optimizer
            if not self._serial_feed_vars:
                self._restore_serial_feed_vars()
            if not self._serial_fetch_vars:
                self._restore_serial_fetch_vars()

            self._init_dist_attr_for_program(no_default)
            # Backup the original distributed information for later restore
            self._original_dist_tensors_for_program = copy.deepcopy(
                self._dist_tensors_for_program
            )
            self._original_dist_ops_for_program = copy.deepcopy(
                self._dist_ops_for_program
            )
            self._tensors_ids = list(self._dist_tensors_for_program.keys())
            self._ops_ids = list(self._dist_ops_for_program.keys())
            self._is_initialized = True

            # TODO: This will be removed in the future
            if with_cpp:
                _copy_dist_attr_to_cpp(self)

            if with_graph:
                set_flags({"FLAGS_convert_all_blocks": True})
                self._serial_graph = IrGraph(
                    core.Graph(self._serial_main_program.desc)
                )
                self._init_dist_attr_for_graph()
                self._need_copy_dist_attr_to_graph = False

        if self._need_copy_dist_attr_to_graph and with_graph:
            self.copy_dist_attr_from_program_to_graph()

    def add_process_mesh(self, process_mesh):
        assert isinstance(
            process_mesh, (ProcessMesh, core.ProcessMesh)
        ), 'The type of dim_mapping must be ProcessMesh.'
        if process_mesh not in self.process_meshes:
            self._process_meshes.append(process_mesh)

    def add_dist_tensor_for_program(self, dist_tensor):
        inner_serial_tensor = dist_tensor.serial_tensor
        inner_serial_tensor_id = inner_serial_tensor.desc.original_id()
        self._dist_tensors_for_program[inner_serial_tensor_id] = dist_tensor

    def add_dist_op_for_program(self, dist_op):
        inner_serial_op = dist_op.serial_op
        inner_serial_op_id = inner_serial_op.desc.original_id()
        self._dist_ops_for_program[inner_serial_op_id] = dist_op

    def get_dist_tensor_for_program(self, serial_tensor):
        serial_tensor_id = serial_tensor.desc.id()
        dist_tensor = self._dist_tensors_for_program.get(serial_tensor_id, None)
        if dist_tensor:
            return dist_tensor
        else:
            serial_tensor_id = serial_tensor.desc.original_id()
            dist_tensor = self._dist_tensors_for_program.get(
                serial_tensor_id, None
            )
            if dist_tensor:
                return dist_tensor
            else:
                return None

    def get_dist_tensor_for_graph(self, serial_tensor_node):
        serial_tensor_node_id = _node_id(serial_tensor_node)
        return self._dist_tensors_for_graph.get(serial_tensor_node_id, None)

    def get_dist_op_for_program(self, serial_op):
        serial_op_id = serial_op.desc.id()
        dist_op = self._dist_ops_for_program.get(serial_op_id, None)
        if dist_op:
            return dist_op
        else:
            serial_op_id = serial_op.desc.original_id()
            dist_op = self._dist_ops_for_program.get(serial_op_id, None)
            if dist_op:
                return dist_op
            else:
                return None

    def del_dist_op_for_program(self, serial_tensor):
        serial_tensor_id = serial_tensor.desc.id()
        if self._dist_ops_for_program.get(serial_tensor_id, None):
            del self._dist_ops_for_program[serial_tensor_id]

    def get_dist_op_for_graph(self, serial_op_node):
        serial_op_node_id = _node_id(serial_op_node)
        return self._dist_ops_for_graph.get(serial_op_node_id, None)

    def get_tensor_dist_attr_for_program(self, serial_tensor):
        serial_tensor_id = serial_tensor.desc.id()
        dist_tensor = self._dist_tensors_for_program.get(serial_tensor_id, None)
        if dist_tensor:
            return dist_tensor.dist_attr
        else:
            serial_tensor_id = serial_tensor.desc.original_id()
            dist_tensor = self._dist_tensors_for_program.get(
                serial_tensor_id, None
            )
            if dist_tensor:
                return dist_tensor.dist_attr
            else:
                return None

    def get_tensor_dist_attr_for_program_with_id(self, tensor_id):
        dist_tensor = self._dist_tensors_for_program.get(tensor_id, None)
        if dist_tensor:
            return dist_tensor.dist_attr
        else:
            return None

    def set_tensor_dist_attr_for_program(self, serial_tensor, dist_attr):
        dist_tensor = DistributedTensor(serial_tensor, dist_attr)
        self.add_dist_tensor_for_program(dist_tensor)

    def get_tensor_dist_attr_for_graph(self, serial_tensor_node):
        serial_tensor_node_id = _node_id(serial_tensor_node)
        dist_tensor = self._dist_tensors_for_graph.get(
            serial_tensor_node_id, None
        )
        if dist_tensor:
            return dist_tensor.dist_attr
        else:
            return None

    def get_op_dist_attr_for_program(self, serial_op):
        serial_op_id = serial_op.desc.id()
        dist_op = self._dist_ops_for_program.get(serial_op_id, None)
        if dist_op:
            return dist_op.dist_attr
        else:
            serial_op_id = serial_op.desc.original_id()
            dist_op = self._dist_ops_for_program.get(serial_op_id, None)
            if dist_op:
                return dist_op.dist_attr
            else:
                return None

    def get_op_dist_attr_for_program_with_id(self, op_id):
        dist_op = self._dist_ops_for_program.get(op_id, None)
        if dist_op:
            return dist_op.dist_attr
        else:
            return None

    def set_op_dist_attr_for_program(self, serial_op, dist_attr):
        dist_op = DistributedOperator(serial_op, dist_attr)
        self.add_dist_op_for_program(dist_op)

    def get_op_dist_attr_for_graph(self, serial_op_node):
        serial_op_node_id = _node_id(serial_op_node)
        dist_op = self._dist_ops_for_graph.get(serial_op_node_id, None)
        if dist_op:
            return dist_op.dist_attr
        else:
            return None

    def get_dist_attr_for_graph(self, serial_node):
        if serial_node.is_var() and serial_node.var() is not None:
            serial_tensor_node_id = _node_id(serial_node)
            dist_tensor = self._dist_tensors_for_graph.get(
                serial_tensor_node_id, None
            )
            if dist_tensor:
                return dist_tensor.dist_attr
            else:
                return None
        if serial_node.is_op() and serial_node.op() is not None:
            serial_op_node_id = _node_id(serial_node)
            dist_op = self._dist_ops_for_graph.get(serial_op_node_id, None)
            if dist_op:
                return dist_op.dist_attr
            else:
                return None
        return None

    def _init_dist_attr_for_program(self, no_default=False):
        # Copy the dist tensors and dist ops annotated by users from the default context
        if not no_default:
            default_ctx = get_default_distributed_context()
            self._process_meshes = copy.deepcopy(default_ctx.process_meshes)
        else:
            default_ctx = self
        # Copy the data parallel flag from the default context
        self._data_parallel = default_ctx.data_parallel
        for block in self._serial_main_program.blocks:
            for tensor in block.vars.values():
                # Copy the distributed tensors in the default context
                default_dist_tensor = default_ctx.get_dist_tensor_for_program(
                    tensor
                )
                if default_dist_tensor and default_ctx is not self:
                    dist_tensor = DistributedTensor(tensor)
                    dist_tensor.dist_attr = copy.deepcopy(
                        default_dist_tensor.dist_attr
                    )
                    self.add_dist_tensor_for_program(dist_tensor)
                current_dist_tensor = self.get_dist_tensor_for_program(tensor)
                if current_dist_tensor is None:
                    dist_tensor = DistributedTensor(tensor)
                    self.add_dist_tensor_for_program(dist_tensor)
            for op in block.ops:
                # Copy the distributed operators in the default context
                default_dist_op = default_ctx.get_dist_op_for_program(op)
                if default_dist_op and default_ctx is not self:
                    dist_op = DistributedOperator(op)
                    dist_op.dist_attr = copy.deepcopy(default_dist_op.dist_attr)
                    self.add_dist_op_for_program(dist_op)
                current_dist_op = self.get_dist_op_for_program(op)
                if current_dist_op is None:
                    dist_op = DistributedOperator(op)
                    self.add_dist_op_for_program(dist_op)
        self._original_dist_tensors_for_program = copy.deepcopy(
            self._dist_tensors_for_program
        )
        self._original_dist_ops_for_program = copy.deepcopy(
            self._dist_ops_for_program
        )

    def _order_nodes_by_program_order(self):
        serial_ordered_tensor_nodes = []
        serial_ordered_op_nodes = []
        all_nodes = []
        visited = {}
        for idx, graph in enumerate(self._serial_graph.all_sub_graphs()):
            for node in graph.all_nodes():
                all_nodes.append(node)
        for node in all_nodes:
            if node.is_var() and node.var() is not None:
                serial_ordered_tensor_nodes.append(node)
                visited[_node_id(node)] = False
            if node.is_op() and node.op() is not None:
                serial_ordered_op_nodes.append(node)
        serial_ordered_tensor_nodes.sort(
            key=lambda node: node.node.original_desc_id()
        )
        serial_ordered_op_nodes.sort(
            key=lambda node: node.node.original_desc_id()
        )
        num_nodes_before = len(serial_ordered_tensor_nodes) + len(
            serial_ordered_op_nodes
        )

        new_serial_ordered_tensor_nodes = []
        new_serial_ordered_op_nodes = []
        new_serial_ordered_nodes = []
        for op_node in serial_ordered_op_nodes:
            tensor_nodes = []
            for tensor_node in op_node.inputs:
                if (
                    tensor_node.is_var()
                    and tensor_node.var() is not None
                    and not visited[_node_id(tensor_node)]
                ):
                    tensor_nodes.append(tensor_node)
                    new_serial_ordered_tensor_nodes.append(tensor_node)
                    visited[_node_id(tensor_node)] = True

            tensor_nodes.sort(key=lambda node: node.node.original_desc_id())
            new_serial_ordered_nodes.extend(tensor_nodes)
            new_serial_ordered_nodes.append(op_node)
            new_serial_ordered_op_nodes.append(op_node)
            tensor_nodes = []
            for tensor_node in op_node.outputs:
                if (
                    tensor_node.is_var()
                    and tensor_node.var() is not None
                    and not visited[_node_id(tensor_node)]
                ):
                    tensor_nodes.append(tensor_node)
                    new_serial_ordered_tensor_nodes.append(tensor_node)
                    visited[_node_id(tensor_node)] = True
            tensor_nodes.sort(key=lambda node: node.node.original_desc_id())
            new_serial_ordered_nodes.extend(tensor_nodes)
        new_serial_ordered_tensor_nodes.sort(
            key=lambda node: node.node.original_desc_id()
        )
        new_serial_ordered_op_nodes.sort(
            key=lambda node: node.node.original_desc_id()
        )
        self._serial_ordered_tensor_nodes = new_serial_ordered_tensor_nodes
        self._serial_ordered_op_nodes = new_serial_ordered_op_nodes
        self._serial_ordered_nodes = new_serial_ordered_nodes
        assert len(self._serial_ordered_nodes) == len(
            self._serial_ordered_tensor_nodes
        ) + len(self._serial_ordered_op_nodes)
        # graph_id -> tensor->name -> node_lists
        self._tensor_nodes_with_same_name = defaultdict(dict)
        for idx, node in enumerate(self._serial_ordered_nodes):
            if node.is_var() and node.var() is not None:
                graph_id = node.node.graph_id()
                tensor_name = node.var().name()
                if (
                    self._tensor_nodes_with_same_name[graph_id].get(
                        tensor_name, None
                    )
                    is None
                ):
                    self._tensor_nodes_with_same_name[graph_id][
                        tensor_name
                    ] = []
                self._tensor_nodes_with_same_name[graph_id][tensor_name].append(
                    (idx, node)
                )

        self._serial_orphan_tensor_nodes = []
        for tensor_node in serial_ordered_tensor_nodes:
            if not visited[_node_id(tensor_node)]:
                self._serial_orphan_tensor_nodes.append(tensor_node)
        if len(self._serial_ordered_nodes) != num_nodes_before:
            print(
                "WARNING: there are some orphan tensors or ops which are not used in the execution."
            )

    def _init_dist_attr_for_graph(self):
        # Convert program to graph and initialize the distributed attributes
        self._order_nodes_by_program_order()
        self._tensor_original_id_to_id = {}
        self._op_original_id_to_id = {}
        for tensor_id, tensor in self._dist_tensors_for_program.items():
            original_id = tensor.serial_tensor.desc.original_id()
            self._tensor_original_id_to_id[original_id] = tensor_id
        for op_id, op in self._dist_ops_for_program.items():
            original_id = op.serial_op.desc.original_id()
            self._op_original_id_to_id[original_id] = op_id
        for node in self.serial_ordered_nodes:
            if node.is_var() and node.var() is not None:
                dist_tensor = None
                tensor_id = node.node.original_desc_id()
                cur_dist_tensor = self._dist_tensors_for_program.get(
                    tensor_id, None
                )
                if cur_dist_tensor is not None:
                    cur_tensor_id = tensor_id
                else:
                    cur_tensor_id = self._tensor_original_id_to_id[tensor_id]
                    cur_dist_tensor = self._dist_tensors_for_program.get(
                        cur_tensor_id, None
                    )
                dist_tensor = cur_dist_tensor
                self._node_id_to_tensor_id[_node_id(node)] = cur_tensor_id
                assert (
                    dist_tensor is not None
                ), "Tensor must have a distributed tensor after the initialization for program."
                serial_tensor_node_id = _node_id(node)
                new_dist_tensor = DistributedTensor(
                    dist_tensor.serial_tensor, dist_tensor.dist_attr
                )
                self._dist_tensors_for_graph[
                    serial_tensor_node_id
                ] = new_dist_tensor
            if node.is_op() and node.op() is not None:
                dist_op = None
                op_id = node.node.original_desc_id()
                cur_dist_op = self._dist_ops_for_program.get(op_id, None)
                if cur_dist_op is not None:
                    cur_op_id = op_id
                else:
                    cur_op_id = self._op_original_id_to_id[op_id]
                    cur_dist_op = self._dist_ops_for_program.get(
                        cur_op_id, None
                    )
                dist_op = cur_dist_op
                self._node_id_to_op_id[_node_id(node)] = cur_op_id
                assert (
                    dist_op is not None
                ), "Operator must have a distributed operator after the initialization for program."
                serial_op_node_id = _node_id(node)
                new_dist_op = DistributedOperator(
                    dist_op.serial_op, dist_op.dist_attr
                )
                self._dist_ops_for_graph[serial_op_node_id] = new_dist_op

    def clear_dist_info_for_program(self):
        self._dist_tensors_for_program.clear()
        self._dist_ops_for_program.clear()

    def clear_dist_info_for_graph(self):
        self._dist_tensors_for_graph.clear()
        self._dist_ops_for_graph.clear()

    def copy_dist_attr_from_program_to_graph(self):
        for node in self.serial_ordered_nodes:
            if node.is_var() and node.var() is not None:
                dist_tensor = None
                tensor_id = node.node.original_desc_id()
                cur_dist_tensor = self._dist_tensors_for_program.get(
                    tensor_id, None
                )
                if cur_dist_tensor is not None:
                    cur_tensor_id = tensor_id
                else:
                    cur_tensor_id = self._tensor_original_id_to_id[tensor_id]
                    cur_dist_tensor = self._dist_tensors_for_program.get(
                        cur_tensor_id, None
                    )
                dist_tensor = cur_dist_tensor
                assert (
                    dist_tensor is not None
                ), "Tensor must have a distributed tensor after the initialization for program."
                serial_tensor_node_id = _node_id(node)
                new_dist_tensor = DistributedTensor(
                    dist_tensor.serial_tensor, dist_tensor.dist_attr
                )
                self._dist_tensors_for_graph[
                    serial_tensor_node_id
                ] = new_dist_tensor
            if node.is_op() and node.op() is not None:
                dist_op = None
                op_id = node.node.original_desc_id()
                cur_dist_op = self._dist_ops_for_program.get(op_id, None)
                if cur_dist_op is not None:
                    cur_op_id = op_id
                else:
                    cur_op_id = self._op_original_id_to_id[op_id]
                    cur_dist_op = self._dist_ops_for_program.get(
                        cur_op_id, None
                    )
                dist_op = cur_dist_op
                assert (
                    dist_op is not None
                ), "Operator must have a distributed operator after the initialization for program."
                serial_op_node_id = _node_id(node)
                new_dist_op = DistributedOperator(
                    dist_op.serial_op, dist_op.dist_attr
                )
                self._dist_ops_for_graph[serial_op_node_id] = new_dist_op

    def copy_dist_attr_from_graph_to_program(self):
        assert (
            self._is_initialized
        ), "Both program and graph must be initialized."
        updated_tensors = {}
        all_nodes = self._serial_ordered_nodes
        process_meshes = [self.process_meshes[0]]
        for node in all_nodes:
            if node.is_var() and node.var() is not None:
                tensor_id = self._node_id_to_tensor_id[_node_id(node)]
                updated = updated_tensors.get(tensor_id, False)
                # If a var has multiples var nodes in graph, only use the first one for now
                if not updated:
                    tensor_dist_attr_for_graph = (
                        self.get_tensor_dist_attr_for_graph(node)
                    )
                    dist_tensor_for_program = self._dist_tensors_for_program[
                        tensor_id
                    ]
                    dist_tensor_for_program.dist_attr = (
                        tensor_dist_attr_for_graph
                    )
                    updated_tensors[tensor_id] = True
                    process_mesh = tensor_dist_attr_for_graph.process_mesh
                    if process_mesh not in process_meshes:
                        process_meshes.append(process_mesh)
            if node.is_op() and node.op() is not None:
                op_id = self._node_id_to_op_id[_node_id(node)]
                op_dist_attr_for_graph = self.get_op_dist_attr_for_graph(node)
                dist_op_for_program = self._dist_ops_for_program[op_id]
                dist_op_for_program.dist_attr = op_dist_attr_for_graph
                process_mesh = op_dist_attr_for_graph.process_mesh
                if process_mesh not in process_meshes:
                    process_meshes.append(process_mesh)
        # NOTE(zhaoyingli):
        # The order of process_meshes is execution order of the ops,
        # which will help pipeline strategy to get pp_rank info.
        self.process_meshes = copy.deepcopy(process_meshes)
        # TODO: the completion algorithm will skipped orphan tensors,
        # here we just set there process_mesh to the first one.
        for orphan_node in self._serial_orphan_tensor_nodes:
            serial_tensor_id = orphan_node.var().id()
            dist_tensor = self._dist_tensors_for_program.get(
                serial_tensor_id, None
            )
            if not dist_tensor:
                serial_tensor_id = orphan_node.var().original_id()
                dist_tensor = self._dist_tensors_for_program.get(
                    serial_tensor_id, None
                )
            dist_tensor.dist_attr.process_mesh = self.process_meshes[0]

    def amend_dist_attr_for_program(self):
        for dist_tensor in self._dist_tensors_for_program.values():
            serial_tensor = dist_tensor.serial_tensor
            dist_attr = dist_tensor.dist_attr
            if serial_tensor.type in __no_shape_var_type__:
                tensor_shape = []
            else:
                tensor_shape = serial_tensor.shape
            dims_mapping = dist_attr.dims_mapping
            process_mesh_shape = dist_attr.process_mesh.shape
            process_mesh_processes = dist_attr.process_mesh.process_ids
            # If the dimension of tensor is less than the sharding dimension of process mesh,
            # we just amend the dimension mapping to -1. (Is this really OK?)
            for i in range(len(tensor_shape)):
                if (
                    dims_mapping[i] != -1
                    and tensor_shape[i] > 0
                    and process_mesh_shape[dims_mapping[i]] > tensor_shape[i]
                ):
                    dims_mapping[i] = -1
                if dims_mapping[i] != -1 and len(process_mesh_processes) == 1:
                    dims_mapping[i] = -1
            dist_attr.dims_mapping = dims_mapping

        for dist_op in self._dist_ops_for_program.values():
            serial_op = dist_op.serial_op
            dist_attr = dist_op.dist_attr
            process_mesh_shape = dist_attr.process_mesh.shape
            process_mesh_processes = dist_attr.process_mesh.process_ids
            for arg_name in serial_op.input_arg_names:
                if dist_op.get_serial_input(arg_name) is None:
                    tensor_shape = []
                else:
                    if (
                        dist_op.get_serial_input(arg_name).type
                        in __no_shape_var_type__
                    ):
                        tensor_shape = []
                    else:
                        tensor_shape = dist_op.get_serial_input(arg_name).shape
                dims_mapping = dist_attr.get_input_dims_mapping(arg_name)
                # If the dimension of tensor is less than the sharding dimension of process mesh,
                # we just amend the dimension mapping to -1. (Is this really OK?)
                for i in range(len(tensor_shape)):
                    if (
                        dims_mapping[i] != -1
                        and tensor_shape[i] > 0
                        and process_mesh_shape[dims_mapping[i]]
                        > tensor_shape[i]
                    ):
                        dims_mapping[i] = -1
                    if (
                        dims_mapping[i] != -1
                        and len(process_mesh_processes) == 1
                    ):
                        dims_mapping[i] = -1
                dist_attr.set_input_dims_mapping(arg_name, dims_mapping)
            for arg_name in serial_op.output_arg_names:
                if (
                    dist_op.get_serial_output(arg_name).type
                    in __no_shape_var_type__
                ):
                    tensor_shape = []
                else:
                    tensor_shape = dist_op.get_serial_output(arg_name).shape
                dims_mapping = dist_attr.get_output_dims_mapping(arg_name)
                # If the dimension of tensor is less than the sharding dimension of process mesh,
                # we just amend the dimension mapping to -1. (Is this really OK?)
                for i in range(len(tensor_shape)):
                    if (
                        dims_mapping[i] != -1
                        and tensor_shape[i] > 0
                        and process_mesh_shape[dims_mapping[i]]
                        > tensor_shape[i]
                    ):
                        dims_mapping[i] = -1
                    if (
                        dims_mapping[i] != -1
                        and len(process_mesh_processes) == 1
                    ):
                        dims_mapping[i] = -1
                dist_attr.set_output_dims_mapping(arg_name, dims_mapping)
            if (
                len(process_mesh_processes) == 1
                and dist_op.serial_op.type != "dropout"
            ):
                dist_op.dist_attr.impl_type = "default"
                dist_op.dist_attr.impl_idx = 0

    def validate_dist_attr_for_program(self):
        if not self._is_initialized:
            raise AssertionError(
                "Program must be initialized before validating its distributed attributes"
            )
        for block in self.serial_main_program.blocks:
            for tensor in block.vars.values():
                dist_tensor = self.get_dist_tensor_for_program(tensor)
                assert (
                    dist_tensor is not None
                ), f"Tensor {dist_tensor.serial_tensor.name} does not have a distributed attribute."
                if (dist_tensor is not None) and (
                    not dist_tensor.validate_dist_attr()
                ):
                    raise AssertionError(
                        f"Tensor {dist_tensor.serial_tensor.name} (id: {dist_tensor.serial_tensor.desc.id()}, original_id: {dist_tensor.serial_tensor.desc.original_id()}) has a wrong distributed attributes {dist_tensor.dist_attr}."
                    )
            for op in block.ops:
                dist_op = self.get_dist_op_for_program(op)
                assert (
                    dist_op is not None
                ), f"Operator {dist_op.serial_op.type} does not have a distributed attribute."
                if (dist_op is not None) and (not dist_op.validate_dist_attr()):
                    raise AssertionError(
                        f"Operator {dist_op.serial_op.type} (id: {dist_op.serial_op.desc.id()}, original_id: {dist_op.serial_op.desc.original_id()}) has a wrong distributed attributes {dist_op.dist_attr} ."
                    )
                if (
                    op.has_attr("op_namescope")
                    and 'auto_parallel/rc_' in op.attr("op_namescope")
                    and not self.strategy.recompute.enable
                ):
                    self.strategy.recompute.enable = True
        return True

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in [
                "_original_serial_main_program",
                "_original_serial_startup_program",
                "_serial_main_program",
                "_serial_startup_program",
                "_serial_graph",
                "_dist_main_programs",
                "_dist_startup_programs",
                "_serial_ordered_nodes",
                "_serial_ordered_tensor_nodes",
                "_serial_ordered_op_nodes",
                "_original_serial_loss",
                "_original_serial_feed_vars",
                "_original_serial_fetch_vars",
                "_serial_loss",
                "_serial_feed_vars",
                "_serial_fetch_vars",
                "_serial_optimizer",
                "_backup_serial_main_program_stack",
                "_backup_serial_startup_program_stack",
                "_pass_context",
                "_tensor_nodes_with_same_name",
            ]:
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))

        # update dist tensor's dist_context
        for key in result._dist_tensors_for_program.keys():
            result._dist_tensors_for_program[key]._dist_context = result
        return result


class DistributedOperatorContext:
    """
    DistributedOperatorContext is used to create a dist op desc in Program.
    Every time to create a new dist op, the context should be updated for it accordingly.
    """

    def __init__(self):
        self._dst_main_program = None
        self._main_block = None
        self._dst_startup_program = None
        self._startup_block = None
        self._cur_src_op = None
        self._cur_dist_attr = None
        self.grad_op_id_to_op_id = {}
        self.grad_var_to_var = defaultdict(dict)
        self._work_block = None
        self.already_init_sync_vars = set()
        self.varname_mapping = None
        self.rank_id = None
        # NOTE Support correct parallelism for high-order differential model.
        # by default exceed_backward_init_op is False and it means we are in Forward phase; After exceed_backward_init_op = True,
        # it means we are in Backward phase.
        # And the final solution should be revise high-order differential logic for these two phases in future.
        self._exceed_backward_init_op = False

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in [
                "_dst_main_program",
                "_dst_startup_program",
                "_cur_src_op",
                "_work_block",
                "_main_block",
                "_startup_block",
            ]:
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    @property
    def dst_main_program(self):
        return self._dst_main_program

    @dst_main_program.setter
    def dst_main_program(self, prog):
        self._dst_main_program = prog
        self._main_block = prog.blocks[0]

    @property
    def main_block(self):
        return self._main_block

    @property
    def dst_startup_program(self):
        return self._dst_startup_program

    @dst_startup_program.setter
    def dst_startup_program(self, prog):
        self._dst_startup_program = prog
        self._startup_block = prog.blocks[0]

    @property
    def startup_block(self):
        return self._startup_block

    @property
    def work_block(self):
        assert self._work_block is not None
        return self._work_block

    @work_block.setter
    def work_block(self, block):
        assert block is not None
        self._work_block = block

    @property
    def cur_src_op(self):
        assert self._cur_src_op is not None
        return self._cur_src_op

    def in_backward_phase(self):
        return self._exceed_backward_init_op

    def prepare_context(self, src_op):
        self._cur_src_op = src_op

        if is_loss_grad_op(src_op):
            self._exceed_backward_init_op = True

        # build input varname mapping
        kinputs = {}
        for input_name in src_op.desc.input_names():
            varnames = []
            for varname in src_op.desc.input(input_name):
                assert varname in self.varname_mapping[src_op.block.idx]
                varnames.append(self.varname_mapping[src_op.block.idx][varname])
            kinputs[input_name] = varnames

        # build output varname mapping
        koutputs = {}
        for output_name in src_op.desc.output_names():
            varnames = []
            for varname in src_op.desc.output(output_name):
                assert varname in self.varname_mapping[src_op.block.idx]
                varnames.append(self.varname_mapping[src_op.block.idx][varname])
            koutputs[output_name] = varnames

        return kinputs, koutputs


class BlockState:
    def __init__(self):
        self.nblock = 0
        self.forward_indices = []
        self.backward_indices = []
        self.backward_to_forward_index_map = {}

    def parse_forward_blocks(self, program):
        program._roll_to_global_block()
        assert program.current_block_idx == 0

        for idx, block in enumerate(program.blocks):
            assert idx == block.idx, "index doesn't match"
            assert (
                block.forward_block_idx == -1
            ), f"forward_block_idx of forward block [{idx}] is not [{block.forward_block_idx}]"
            self.forward_indices.append(idx)
            self.nblock += 1

        assert self.nblock >= 1

    def parse_backward_blocks(self, program):
        assert (
            0 in self.forward_indices
        ), f"forward block idx are{self.forward_indices}"
        self.backward_to_forward_index_map[0] = 0

        for idx, block in enumerate(program.blocks):
            if idx < len(self.forward_indices):
                continue

            assert idx == block.idx, "index doesn't match"
            assert block.forward_block_idx in self.forward_indices
            self.backward_indices.append(idx)
            self.backward_to_forward_index_map[idx] = block.forward_block_idx
            self.nblock += 1

        assert self.nblock == len(program.blocks)


class UpDownStream:
    def __init__(self):
        self._ups = {}
        self._downs = {}

    def add_up_stream(self, rank, up_stream):
        ups = self._ups.get(rank, None)
        if not ups:
            self._ups[rank] = [up_stream]
        elif up_stream != -1:
            ups = list(filter(lambda a: a != -1, ups))
            ups.append(up_stream)
            self._ups[rank] = ups

    def add_down_stream(self, rank, down_stream):
        downs = self._downs.get(rank, None)
        if not downs:
            self._downs[rank] = [down_stream]
        elif down_stream != -1:
            downs = list(filter(lambda a: a != -1, downs))
            downs.append(down_stream)
            self._downs[rank] = downs

    def add_pair_stream(self, up, down):
        self.add_up_stream(up, -1)
        self.add_up_stream(down, up)
        self.add_down_stream(up, down)
        self.add_down_stream(down, -1)

    def ups(self, rank):
        ups = self._ups.get(rank, None)
        if not ups:
            return None
        return list(set(ups))

    def downs(self, rank):
        downs = self._downs.get(rank, None)
        if not downs:
            return None
        return list(set(downs))
