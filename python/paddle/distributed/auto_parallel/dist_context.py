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
from paddle.fluid import framework
from paddle.fluid import core
from .dist_attribute import TensorDistributedAttribute
from .dist_attribute import OperatorDistributedAttribute
from .dist_tensor import DistributedTensor
from .dist_op import DistributedOperator
from .process_mesh import ProcessMesh

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


class DistributedContext:
    """
    DistributedContext is used to collect related distributed information for program and graph.
    One auto-parallel run should use its own DistributedContext to avoid interfering other run.
    """

    def __init__(self, program=None):
        # Program related data members
        self._serial_program = program
        self._is_initialized_for_program = False
        self._dist_tensors_for_program = {}
        self._dist_ops_for_program = {}
        # Graph related data members
        self._is_initialized_for_graph = False
        self._serial_graph = None
        self._dist_tensors_for_graph = {}
        self._dist_ops_for_graph = {}
        self._node_id_to_tensor_id = {}
        self._node_id_to_op_id = {}
        # Other data members
        self._dist_op_context = DistributedOperatorContext()
        self._process_meshes = []
        self._serial_ordered_nodes = []
        self._tensor_id_to_tensor_node_ids = {}

        # Distributed programs
        self._dist_main_programs = {}
        self._dist_startup_programs = {}

    @property
    def serial_program(self):
        return self._serial_program

    @property
    def serial_graph(self):
        return self._serial_graph

    @serial_program.setter
    def serial_program(self, program):
        assert self._serial_program is None, \
            "This distributed context has already been realted to a serial program"
        self._serial_program = program

    @property
    def serial_ordered_nodes(self):
        return self._serial_ordered_nodes

    @property
    def process_meshes(self):
        return self._process_meshes

    @property
    def dist_op_context(self):
        return self._dist_op_context

    @property
    def dist_main_programs(self):
        return self._dist_main_programs

    @property
    def dist_startup_programs(self):
        return self._dist_startup_programs

    def add_process_mesh(self, process_mesh):
        assert isinstance(process_mesh, ProcessMesh), \
            'The type of dim_mapping must be ProcessMesh.'
        if process_mesh not in self.process_meshes:
            self._process_meshes.append(process_mesh)

    def add_dist_tensor_for_program(self, dist_tensor):
        inner_serial_tensor = dist_tensor.serial_tensor
        inner_serial_tensor_id = inner_serial_tensor.desc.id()
        self._dist_tensors_for_program[inner_serial_tensor_id] = dist_tensor

    def add_dist_op_for_program(self, dist_op):
        inner_serial_op = dist_op.serial_op
        inner_serial_op_id = inner_serial_op.desc.id()
        self._dist_ops_for_program[inner_serial_op_id] = dist_op

    def get_dist_tensor_for_program(self, serial_tensor):
        serial_tensor_id = serial_tensor.desc.id()
        dist_tensor = self._dist_tensors_for_program.get(serial_tensor_id, None)
        if dist_tensor:
            return dist_tensor
        else:
            serial_tensor_id = serial_tensor.desc.original_id()
            dist_tensor = self._dist_tensors_for_program.get(serial_tensor_id,
                                                             None)
            if dist_tensor:
                return dist_tensor
            else:
                return None

    def get_dist_tensor_for_graph(self, serial_tensor_node):
        serial_tensor_node_id = serial_tensor_node.id()
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
        serial_op_node_id = serial_op_node.id()
        return self._dist_ops_for_graph.get(serial_op_node_id, None)

    def get_tensor_dist_attr_for_program(self, serial_tensor):
        serial_tensor_id = serial_tensor.desc.id()
        dist_tensor = self._dist_tensors_for_program.get(serial_tensor_id, None)
        if dist_tensor:
            return dist_tensor.dist_attr
        else:
            serial_tensor_id = serial_tensor.desc.original_id()
            dist_tensor = self._dist_tensors_for_program.get(serial_tensor_id,
                                                             None)
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
        serial_tensor_node_id = serial_tensor_node.id()
        dist_tensor = self._dist_tensors_for_graph.get(serial_tensor_node_id,
                                                       None)
        if dist_tensor:
            return dist_tensor.dist_attr
        else:
            return None

    # def set_tensor_dist_attr_for_graph(self, serial_tensor_node, dist_attr):
    #     assert serial_tensor_node.is_var() and \
    #         serial_tensor_node.var() is not None
    #     serial_tensor_id = serial_tensor_node.node.original_desc_id()
    #     dist_tensor = self._dist_tensors_for_program.get(serial_tensor_id, None)
    #     assert dist_tensor is not None, \
    #         "The distributed tensor of the program has not been added to this context."
    #     serial_tensor_node_id = serial_tensor_node.id()
    #     new_dist_tensor = DistributedTensor(dist_tensor.serial_tensor,
    #                                         dist_attr)
    #     self._dist_tensors_for_graph[serial_tensor_node_id] = new_dist_tensor

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
        serial_op_node_id = serial_op_node.id()
        dist_op = self._dist_ops_for_graph.get(serial_op_node_id, None)
        if dist_op:
            return dist_op.dist_attr
        else:
            return None

    # def set_op_dist_attr_for_graph(self, serial_op_node, dist_attr):
    #     assert serial_op_node.is_op() and \
    #         serial_op_node.op() is not None
    #     serial_op_id = serial_op_node.node.original_desc_id()
    #     dist_op = self._dist_ops_for_program.get(serial_op_id, None)
    #     assert dist_op is not None, \
    #         "The distributed operator of the program has not been added to this context."
    #     serial_op_node_id = serial_op_node.id()
    #     new_dist_op = DistributedOperator(dist_op.serial_op, dist_attr)
    #     self._dist_ops_for_graph[serial_op_node_id] = new_dist_op

    def get_dist_attr_for_graph(self, serial_node):
        if serial_node.is_var() and serial_node.var() is not None:
            serial_tensor_node_id = serial_node.id()
            dist_tensor = self._dist_tensors_for_graph.get(
                serial_tensor_node_id, None)
            if dist_tensor:
                return dist_tensor.dist_attr
            else:
                return None
        if serial_node.is_op() and serial_node.op() is not None:
            serial_op_node_id = serial_node.id()
            dist_op = self._dist_ops_for_graph.get(serial_op_node_id, None)
            if dist_op:
                return dist_op.dist_attr
            else:
                return None
        return None

    def init_dist_attr_for_program(self):
        assert self._serial_program, \
            "Please set the program of this context before initializing its distribute attributes."
        if self._is_initialized_for_program:
            return
        # Copy the dist tensors and dist ops annotated by users from the default context
        default_ctx = get_default_distributed_context()
        self._process_meshes = copy.deepcopy(default_ctx.process_meshes)
        for block in self._serial_program.blocks:
            for tensor in block.vars.values():
                # Copy the distributed tensors in the default context
                default_dist_tensor = default_ctx.get_dist_tensor_for_program(
                    tensor)
                if default_dist_tensor and default_ctx is not self:
                    self.add_dist_tensor_for_program(default_dist_tensor)
                current_dist_tensor = self.get_dist_tensor_for_program(tensor)
                if current_dist_tensor is None:
                    dist_tensor = DistributedTensor(tensor)
                    self.add_dist_tensor_for_program(dist_tensor)
            for op in block.ops:
                # Copy the distributed operators in the default context
                default_dist_op = default_ctx.get_dist_op_for_program(op)
                if default_dist_op and default_ctx is not self:
                    self.add_dist_op_for_program(default_dist_op)
                current_dist_op = self.get_dist_op_for_program(op)
                if current_dist_op is None:
                    dist_op = DistributedOperator(op)
                    self.add_dist_op_for_program(dist_op)
        self._is_initialized_for_program = True

    def order_nodes_by_program_order(self):
        def _contains(nodes, target_node):
            for node in nodes:
                if node.id() == target_node.id():
                    return True
            return False

        ordered_tensor_nodes = []
        ordered_op_nodes = []
        all_nodes = self._serial_graph.all_nodes()
        for node in all_nodes:
            if node.is_var() and node.var() is not None:
                ordered_tensor_nodes.append(node)
            if node.is_op() and node.op() is not None:
                ordered_op_nodes.append(node)
        ordered_tensor_nodes.sort(key=lambda node: node.node.original_desc_id())
        ordered_op_nodes.sort(key=lambda node: node.node.original_desc_id())
        for op_node in ordered_op_nodes:
            tensor_nodes = []
            for tensor_node in op_node.inputs:
                if tensor_node.is_var() \
                    and tensor_node.var() is not None \
                    and not _contains(self._serial_ordered_nodes, tensor_node):
                    tensor_nodes.append(tensor_node)
            tensor_nodes.sort(key=lambda node: node.node.original_desc_id())
            self._serial_ordered_nodes.extend(tensor_nodes)
            self._serial_ordered_nodes.append(op_node)
            tensor_nodes = []
            for tensor_node in op_node.outputs:
                if tensor_node.is_var() \
                    and tensor_node.var() is not None \
                    and not _contains(self._serial_ordered_nodes, tensor_node):
                    tensor_nodes.append(tensor_node)
            self._serial_ordered_nodes.extend(tensor_nodes)
        num_nodes_before = len(ordered_tensor_nodes) + len(ordered_op_nodes)
        assert len(self._serial_ordered_nodes) == num_nodes_before, \
            "The number of nodes before ordering is not the same after ordering."

    def init_dist_attr_for_graph(self):
        assert self._is_initialized_for_program, \
            "The program must be initialized before initializing the distributed attributes for its graph."
        if self._is_initialized_for_graph:
            return
        # Convert program to graph
        self._serial_graph = framework.IrGraph(
            core.Graph(self._serial_program.desc))
        all_nodes = self._serial_graph.all_nodes()
        self.order_nodes_by_program_order()
        for node in self.serial_ordered_nodes:
            if node.is_var() and node.var() is not None:
                dist_tensor = None
                tensor_id = node.node.original_desc_id()
                for cur_tensor_id, cur_dist_tensor in self._dist_tensors_for_program.items(
                ):
                    if tensor_id == cur_tensor_id \
                        or tensor_id == cur_dist_tensor.serial_tensor.desc.original_id():
                        dist_tensor = cur_dist_tensor
                        self._node_id_to_tensor_id[node.id()] = cur_tensor_id
                assert dist_tensor is not None, \
                    "Tensor must have a distributed tensor after the initialization for program."
                serial_tensor_node_id = node.id()
                new_dist_tensor = DistributedTensor(dist_tensor.serial_tensor,
                                                    dist_tensor.dist_attr)
                self._dist_tensors_for_graph[
                    serial_tensor_node_id] = new_dist_tensor
            if node.is_op() and node.op() is not None:
                dist_op = None
                op_id = node.node.original_desc_id()
                for cur_op_id, cur_dist_op in self._dist_ops_for_program.items(
                ):
                    if op_id == cur_op_id \
                        or op_id == cur_dist_op.serial_op.desc.original_id():
                        dist_op = cur_dist_op
                        self._node_id_to_op_id[node.id()] = cur_op_id
                assert dist_op is not None, \
                    "Operator must have a distributed operator after the initialization for program."
                serial_op_node_id = node.id()
                new_dist_op = DistributedOperator(dist_op.serial_op,
                                                  dist_op.dist_attr)
                self._dist_ops_for_graph[serial_op_node_id] = new_dist_op
        self._is_initialized_for_graph = True

    def clear_dist_info_for_program(self):
        self._dist_tensors_for_program.clear()
        self._dist_ops_for_program.clear()

    def clear_dist_info_for_graph(self):
        self._dist_tensors_for_graph.clear()
        self._dist_ops_for_graph.clear()

    def copy_dist_attr_from_graph_to_program(self):
        assert self._is_initialized_for_program and self._is_initialized_for_graph, \
            "Both program and graph must be initialized."
        updated_tensors = {}
        all_nodes = self._serial_graph.all_nodes()
        for node in all_nodes:
            if node.is_var() and node.var() is not None:
                tensor_id = self._node_id_to_tensor_id[node.id()]
                updated = updated_tensors.get(tensor_id, False)
                # If a var has multiples var nodes in graph, only use the first one for now
                if not updated:
                    tensor_dist_attr_for_graph = self.get_tensor_dist_attr_for_graph(
                        node)
                    dist_tensor_for_program = self._dist_tensors_for_program[
                        tensor_id]
                    dist_tensor_for_program.dist_attr = tensor_dist_attr_for_graph
                    updated_tensors[tensor_id] = True
            if node.is_op() and node.op() is not None:
                op_id = self._node_id_to_op_id[node.id()]
                op_dist_attr_for_graph = self.get_op_dist_attr_for_graph(node)
                dist_op_for_program = self._dist_ops_for_program[op_id]
                dist_op_for_program.dist_attr = op_dist_attr_for_graph

    def amend_dist_attr_for_program(self):
        for dist_tensor in self._dist_tensors_for_program.values():
            serial_tensor = dist_tensor.serial_tensor
            dist_attr = dist_tensor.dist_attr
            if serial_tensor.type == core.VarDesc.VarType.READER:
                tensor_shape = []
            else:
                tensor_shape = serial_tensor.shape
            dims_mapping = dist_attr.dims_mapping
            process_mesh_shape = dist_attr.process_mesh.topology
            # If the dimension of tensor is less than the sharding dimension of process mesh,
            # we just amend the dimension mapping to -1. (Is this really OK?)
            for i in range(len(tensor_shape)):
                if dims_mapping[i] != -1 and tensor_shape[i] > 0 \
                    and process_mesh_shape[dims_mapping[i]] > tensor_shape[i]:
                    dims_mapping[i] = -1

        for dist_op in self._dist_ops_for_program.values():
            serial_op = dist_op.serial_op
            dist_attr = dist_op.dist_attr
            for arg_name in serial_op.input_arg_names:
                if dist_op.get_serial_input(arg_name) is None:
                    tensor_shape = []
                else:
                    if dist_op.get_serial_input(arg_name).type == core.VarDesc.VarType.READER \
                        or dist_op.serial_op.type == "create_py_reader":
                        tensor_shape = []
                    else:
                        tensor_shape = dist_op.get_serial_input(arg_name).shape
                dims_mapping = dist_attr.get_input_dims_mapping(arg_name)
                process_mesh_shape = dist_attr.process_mesh.topology
                # If the dimension of tensor is less than the sharding dimension of process mesh,
                # we just amend the dimension mapping to -1. (Is this really OK?)
                for i in range(len(tensor_shape)):
                    if dims_mapping[i] != -1 and tensor_shape[i] > 0 \
                        and process_mesh_shape[dims_mapping[i]] > tensor_shape[i]:
                        dims_mapping[i] = -1
            for arg_name in serial_op.output_arg_names:
                if dist_op.get_serial_output(
                        arg_name).type == core.VarDesc.VarType.READER:
                    tensor_shape = []
                else:
                    tensor_shape = dist_op.get_serial_output(arg_name).shape
                dims_mapping = dist_attr.get_output_dims_mapping(arg_name)
                process_mesh_shape = dist_attr.process_mesh.topology
                # If the dimension of tensor is less than the sharding dimension of process mesh,
                # we just amend the dimension mapping to -1. (Is this really OK?)
                for i in range(len(tensor_shape)):
                    if dims_mapping[i] != -1 and tensor_shape[i] > 0 \
                        and process_mesh_shape[dims_mapping[i]] > tensor_shape[i]:
                        dims_mapping[i] = -1

    def validate_dist_attr_for_program(self):
        if not self._is_initialized_for_program:
            assert False, \
                "Program must be initialized before validating its distributed attributes"
        for block in self.serial_program.blocks:
            for tensor in block.vars.values():
                dist_tensor = self.get_dist_tensor_for_program(tensor)
                if (dist_tensor is not None) and (
                        not dist_tensor.validate_dist_attr()):
                    assert False, "Tensor {} has a wrong distributed attributes {}.".format(
                        dist_tensor.serial_tensor.name, dist_tensor.dist_attr)
            for op in block.ops:
                dist_op = self.get_dist_op_for_program(op)
                if (dist_op is not None) and (not dist_op.validate_dist_attr()):
                    assert False, "Operator {} has a wrong distributed attributes {}.".format(
                        dist_op.serial_op.type, dist_tensor.dist_attr)
        return True

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "_serial_program" or k == "_serial_graph" \
                or k == "_dist_main_programs" or k == "_dist_startup_programs" \
                or k == "_serial_ordered_nodes":
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
        self._dst_startup_program = None
        self._varname_mapping = None
        self._rank_id = None
        self._cur_src_op = None
        self._cur_dist_attr = None
        self.grad_op_id_to_op_id = {}
        self.already_init_sync_vars = set()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "_dst_main_program" or k == "_dst_startup_program" or k == "_cur_src_op":
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def set_dst_main_program(self, prog):
        self._dst_main_program = prog

    def get_dst_main_program(self):
        return self._dst_main_program

    def set_dst_startup_program(self, prog):
        self._dst_startup_program = prog

    def get_dst_startup_program(self):
        return self._dst_startup_program

    def set_varname_mapping(self, mapping):
        self._varname_mapping = mapping

    def get_varname_mapping(self):
        return self._varname_mapping

    def set_rank_id(self, rank_id):
        self._rank_id = rank_id

    def get_rank_id(self):
        return self._rank_id

    def set_cur_src_op(self, cur_src_op):
        self._cur_src_op = cur_src_op

    def get_cur_src_op(self):
        return self._cur_src_op

    def prepare_context(self, src_op):

        self.set_cur_src_op(src_op)

        # build input varname mapping
        kinputs = {}
        for input_name in src_op.desc.input_names():
            varnames = []
            for varname in src_op.desc.input(input_name):
                assert varname in self._varname_mapping
                varnames.append(self._varname_mapping[varname])
            kinputs[input_name] = varnames

        # build output varname mapping
        koutputs = {}
        for output_name in src_op.desc.output_names():
            varnames = []
            for varname in src_op.desc.output(output_name):
                assert varname in self._varname_mapping
                varnames.append(self._varname_mapping[varname])
            koutputs[output_name] = varnames

        return kinputs, koutputs
