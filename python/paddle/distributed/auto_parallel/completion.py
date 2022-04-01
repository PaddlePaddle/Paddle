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

import copy
from copy import deepcopy
import time

from paddle.fluid import core
from paddle.fluid import framework

from .utils import print_program_with_dist_attr
from .operators import find_best_compatible_distributed_operator_impl
from .dist_context import get_default_distributed_context, _node_id
from .dist_tensor import DistributedTensor
from .dist_op import DistributedOperator
from .dist_attribute import TensorDistributedAttribute
from .dist_attribute import OperatorDistributedAttribute
from .process_mesh import ProcessMesh
from paddle.distributed.fleet.meta_optimizers.common import OpRole


def compute_compatible_process_mesh(process_mesh_list):
    """Compute the compatible process mesh given a list of process meshes."""
    if not process_mesh_list:
        return None

    def _compute_compatible_process_mesh_two(pm1, pm2):
        if pm1 is None:
            return True, pm2
        if pm2 is None:
            return True, pm1
        if pm1 == pm2:
            return True, pm1
        if pm1.processes == pm2.processes:
            if len(pm1.topology) >= len(pm2.topology):
                return True, pm1
            else:
                return True, pm2
        process_set1 = set(pm1.processes)
        process_set2 = set(pm2.processes)
        if process_set1.issubset(process_set2):
            return True, pm2
        if process_set2.issubset(process_set1):
            return True, pm1
        return False, None

    compatible_result = None
    for process_mesh in process_mesh_list:
        compatible, compatible_result = _compute_compatible_process_mesh_two(
            compatible_result, process_mesh)
        if not compatible:
            return None
    return copy.deepcopy(compatible_result)


def compute_compatible_dim_mapping(dim_mapping_list):
    """Compute the compatible dim mapping given a list of dim mapping."""
    if not dim_mapping_list:
        return None

    def _compute_compatible_dim_mapping_two(dm1, dm2):
        if dm1 == -1:
            return True, dm2
        if dm2 == -1:
            return True, dm1
        if dm1 == dm2:
            return True, dm1
        return False, None

    compatible_result = -1
    for mapping in dim_mapping_list:
        compatible, compatible_result = _compute_compatible_dim_mapping_two(
            compatible_result, mapping)
        if not compatible:
            return None
    return compatible_result


def compute_compatible_dims_mapping(dims_mapping_list):
    """Compute the compatible dims mapping given a list of dims mapping.
       Each of dims mapping is also a list.
    """
    if not dims_mapping_list:
        return None
    length = len(dims_mapping_list[0])
    for dims_mapping in dims_mapping_list:
        if dims_mapping is None:
            return None
        if len(dims_mapping) != length:
            return None
    compatible_result = []
    for dim_mappings in zip(*dims_mapping_list):
        compatible_dim_mapping = compute_compatible_dim_mapping(
            list(dim_mappings))
        if compatible_dim_mapping is None:
            return None
        compatible_result.append(compatible_dim_mapping)
    return compatible_result


def merge_process_mesh_two(pm1, pm2):
    process_set1 = set()
    process_set2 = set()
    if pm1 is None and pm2 is None:
        return None
    if pm1 is not None:
        process_set1 = set(pm1.processes)
    if pm2 is not None:
        process_set2 = set(pm2.processes)
    merged_process_set = process_set1.union(process_set2)
    merged_process_mesh = ProcessMesh(list(merged_process_set))
    return merged_process_mesh


class Completer:
    def __init__(self, dist_context):
        assert dist_context is not None
        self._dist_context = dist_context

    def _update_tensor_node_dims_mapping(self, tensor_node, fwd=True):
        changed = False
        if (not tensor_node.is_var()) or (tensor_node.var() is None):
            return False
        tensor_desc = tensor_node.var()
        # Skip reader tensor
        if tensor_desc.type() == core.VarDesc.VarType.READER \
            or tensor_desc.type == core.VarDesc.VarType.LOD_TENSOR_ARRAY \
            or tensor_desc.type == core.VarDesc.VarType.STEP_SCOPES:
            return False
        tensor_dist_attr = self._dist_context.get_tensor_dist_attr_for_graph(
            tensor_node)
        assert tensor_dist_attr is not None
        if tensor_dist_attr.is_annotated("dims_mapping"):
            return False
        tensor_dims_mapping = tensor_dist_attr.dims_mapping
        if fwd:
            dims_mapping_list = []
            for pred_op_node in tensor_node.inputs:
                if pred_op_node.op() is not None:
                    if pred_op_node.op().type() == "create_py_reader" \
                        or pred_op_node.op().type() == "create_double_buffer_reader" \
                        or pred_op_node.op().type() == "read":
                        continue
                    op_dist_attr = self._dist_context.get_op_dist_attr_for_graph(
                        pred_op_node)
                    if op_dist_attr.process_mesh == tensor_dist_attr.process_mesh:
                        op_dims_mapping = op_dist_attr.get_output_dims_mapping(
                            tensor_desc.name())
                        dims_mapping_list.append(op_dims_mapping)
            dims_mapping_list.append(tensor_dims_mapping)
            compatible_dims_mapping = compute_compatible_dims_mapping(
                dims_mapping_list)
            if (compatible_dims_mapping is not None) and \
                (compatible_dims_mapping != tensor_dims_mapping):
                tensor_dist_attr.dims_mapping = compatible_dims_mapping
                changed = True
        else:
            dims_mapping_list = []
            for succ_op_node in tensor_node.outputs:
                if succ_op_node.op() is not None:
                    if succ_op_node.op().type() == "create_py_reader" \
                        or succ_op_node.op().type() == "create_double_buffer_reader" \
                        or succ_op_node.op().type() == "read":
                        continue
                    op_dist_attr = self._dist_context.get_op_dist_attr_for_graph(
                        succ_op_node)
                    if op_dist_attr.process_mesh == tensor_dist_attr.process_mesh:
                        op_dims_mapping = op_dist_attr.get_input_dims_mapping(
                            tensor_desc.name())
                        dims_mapping_list.append(op_dims_mapping)
            dims_mapping_list.append(tensor_dims_mapping)
            compatible_dims_mapping = compute_compatible_dims_mapping(
                dims_mapping_list)
            if (compatible_dims_mapping is not None) and \
                (compatible_dims_mapping != tensor_dims_mapping):
                tensor_dist_attr.dims_mapping = compatible_dims_mapping
                changed = True
        return changed

    def _update_op_node_dims_mapping(self, op_node, fwd=True):
        changed = False
        if (not op_node.is_op()) or (op_node.op() is None):
            return False
        # Skip reader op
        op_desc = op_node.op()
        if op_desc.type() == "create_py_reader" \
            or op_desc.type() == "create_double_buffer_reader" \
            or op_desc.type() == "read":
            return False
        dist_op = self._dist_context.get_dist_op_for_graph(op_node)
        op_dist_attr = dist_op.dist_attr
        if fwd:
            for tensor_node in op_node.inputs:
                if tensor_node.is_var() and tensor_node.var() is not None:
                    if tensor_node.var().type() == core.VarDesc.VarType.READER:
                        continue
                    tensor_desc = tensor_node.var()
                    if op_dist_attr.is_annotated_input_dims_mapping(
                            tensor_desc.name()):
                        continue
                    tensor_dist_attr = self._dist_context.get_tensor_dist_attr_for_graph(
                        tensor_node)
                    if op_dist_attr.process_mesh == tensor_dist_attr.process_mesh:
                        tensor_dims_mapping = tensor_dist_attr.dims_mapping
                        op_dims_mapping = op_dist_attr.get_input_dims_mapping(
                            tensor_desc.name())
                        compatible_dims_mapping = compute_compatible_dims_mapping(
                            [op_dims_mapping, tensor_dims_mapping])
                        if (compatible_dims_mapping is not None) and \
                            (compatible_dims_mapping != op_dims_mapping):
                            op_dist_attr.set_input_dims_mapping(
                                tensor_desc.name(), compatible_dims_mapping)
                            changed = True
            # Find the most compatible implemenetations from the distributed operator
            op_dist_impl = find_best_compatible_distributed_operator_impl(
                dist_op, fwd=True)
            if op_dist_impl is not None:
                dim_changed = op_dist_impl.update_dims_mapping(dist_op)
                if dim_changed:
                    changed = True
                if op_dist_impl.is_auto_compatible(dist_op):
                    if op_dist_impl.type == "elementwise":
                        op_dist_attr.impl_type = "default"
                    else:
                        op_dist_attr.impl_type = op_dist_impl.type
                    op_dist_attr.impl_idx = op_dist_impl.idx
        else:
            for tensor_node in op_node.outputs:
                if tensor_node.is_var() and tensor_node.var() is not None:
                    if tensor_node.var().type() == core.VarDesc.VarType.READER:
                        continue
                    tensor_desc = tensor_node.var()
                    if op_dist_attr.is_annotated_output_dims_mapping(
                            tensor_desc.name()):
                        continue
                    tensor_dist_attr = self._dist_context.get_tensor_dist_attr_for_graph(
                        tensor_node)
                    if op_dist_attr.process_mesh == tensor_dist_attr.process_mesh:
                        tensor_dims_mapping = tensor_dist_attr.dims_mapping
                        op_dims_mapping = op_dist_attr.get_output_dims_mapping(
                            tensor_desc.name())
                        compatible_dims_mapping = compute_compatible_dims_mapping(
                            [op_dims_mapping, tensor_dims_mapping])
                        if (compatible_dims_mapping is not None) and \
                            (compatible_dims_mapping != op_dims_mapping):
                            op_dist_attr.set_output_dims_mapping(
                                tensor_desc.name(), compatible_dims_mapping)
                            changed = True
            # Find the most compatible implemenetations from the distributed operator
            op_dist_impl = find_best_compatible_distributed_operator_impl(
                dist_op, fwd=False)
            if op_dist_impl is not None:
                dim_changed = op_dist_impl.update_dims_mapping(dist_op)
                if dim_changed:
                    changed = True
                if op_dist_impl.is_auto_compatible(dist_op):
                    if op_dist_impl.type == "elementwise":
                        op_dist_attr.impl_type = "default"
                    else:
                        op_dist_attr.impl_type = op_dist_impl.type
                    op_dist_attr.impl_idx = op_dist_impl.idx
        return changed

    def _update_dims_mapping_between_graphs(self):
        changed = False
        for parent_node, child_node in self._node_pairs_between_graphs:
            parent_node_dist_attr = self._dist_context.get_dist_attr_for_graph(
                parent_node)
            child_node_dist_attr = self._dist_context.get_dist_attr_for_graph(
                child_node)
            parent_node_dims_mapping = parent_node_dist_attr.dims_mapping
            child_node_dims_mapping = child_node_dist_attr.dims_mapping
            compatible_dims_mapping = compute_compatible_dims_mapping(
                [parent_node_dims_mapping, child_node_dims_mapping])
            if (compatible_dims_mapping is not None) \
                and (compatible_dims_mapping != parent_node_dims_mapping):
                parent_node_dist_attr.dims_mapping = compatible_dims_mapping
                changed = True
            if (compatible_dims_mapping is not None) \
                and (compatible_dims_mapping != child_node_dims_mapping):
                parent_node_dist_attr.dims_mapping = compatible_dims_mapping
                changed = True
        return changed

    def _update_dims_mapping(self):
        # Complete dims_mapping for each node
        reach_fix_point = False
        while not reach_fix_point:
            changed = False
            for is_fwd in [True, False]:
                all_nodes = self._dist_context.serial_ordered_nodes \
                    if is_fwd else reversed(self._dist_context.serial_ordered_nodes)
                for node in all_nodes:
                    if node.is_var() and node.var() is not None:
                        tensor_changed = self._update_tensor_node_dims_mapping(
                            node, fwd=is_fwd)
                        if tensor_changed:
                            changed = True
                    if node.is_op() and node.op() is not None:
                        op_changed = self._update_op_node_dims_mapping(
                            node, fwd=is_fwd)
                        if op_changed:
                            changed = True
                graph_changed = self._update_dims_mapping_between_graphs()
                if graph_changed:
                    changed = True
            if changed:
                reach_fix_point = False
            else:
                reach_fix_point = True

    def _update_process_mesh_by_nearest(self, op_node, nearest_op_node):
        op_dist_attr = self._dist_context.get_dist_attr_for_graph(op_node)
        # Set the process mesh of the op node by its nearest op node
        if not op_dist_attr.is_annotated("process_mesh"):
            process_mesh = op_dist_attr.process_mesh
            nearest_op_dis_attr = self._dist_context.get_dist_attr_for_graph(
                nearest_op_node)
            nearest_process_mesh = nearest_op_dis_attr.process_mesh
            compatible_process_mesh = compute_compatible_process_mesh(
                [process_mesh, nearest_process_mesh])
            if compatible_process_mesh is not None \
                and process_mesh != compatible_process_mesh:
                op_dist_attr.process_mesh = compatible_process_mesh
        # Skip the process_mesh setting of inputs and outputs of while_op
        if op_dist_attr.op_type == "while":
            return
        # Set the process mesh of the op node's leaf-inputs
        for tensor_node in op_node.inputs:
            if tensor_node.is_var() and tensor_node.var() is not None:
                tensor_dist_attr = self._dist_context.get_tensor_dist_attr_for_graph(
                    tensor_node)
                if tensor_dist_attr.is_annotated("process_mesh"):
                    continue
                # Skip the non-leaf var node
                if len(tensor_node.inputs) != 0:
                    continue
                compatible_process_mesh = compute_compatible_process_mesh(
                    [tensor_dist_attr.process_mesh, op_dist_attr.process_mesh])
                if compatible_process_mesh is not None \
                    and tensor_dist_attr.process_mesh != compatible_process_mesh:
                    tensor_dist_attr.process_mesh = compatible_process_mesh
        # Set the process mesh of the op node's outputs
        for tensor_node in op_node.outputs:
            if tensor_node.is_var() and tensor_node.var() is not None:
                tensor_dist_attr = self._dist_context.get_tensor_dist_attr_for_graph(
                    tensor_node)
                if tensor_dist_attr.is_annotated("process_mesh"):
                    continue
                compatible_process_mesh = compute_compatible_process_mesh(
                    [tensor_dist_attr.process_mesh, op_dist_attr.process_mesh])
                if compatible_process_mesh is not None \
                    and tensor_dist_attr.process_mesh != compatible_process_mesh:
                    tensor_dist_attr.process_mesh = compatible_process_mesh

    def _update_process_mesh_for_specials(self):
        def _find_nearest_tensor_node_before(nodes, idx, var_name):
            for node in reversed(nodes[:idx]):
                if node.is_var() and node.var() is not None \
                    and node.var().name() == var_name:
                    return node

        def _find_nearest_tensor_node_after(nodes, idx, var_name):
            for node in nodes[idx + 1:]:
                if node.is_var() and node.var() is not None \
                    and node.var().name() == var_name:
                    return node

        def _find_nodes_related_to_cond(source_node):
            related_nodes = []
            visited = set()
            frontier = list()
            frontier.append(source_node)
            # BFS
            while len(frontier) != 0:
                cur = frontier[0]
                frontier = frontier[1:]
                if _node_id(cur) in visited:
                    continue
                # TODO: need more restrictions
                for node in cur.inputs:
                    if node.is_var() and node.var() is not None:
                        if node.var().type() != core.VarDesc.VarType.READER \
                            and len(node.var().shape()) == 1:
                            frontier.append(node)
                            related_nodes.append(node)
                    if node.is_op() and node.op() is not None:
                        flag = True
                        if node.op().type() == "create_py_reader" \
                            or node.op().type() == "create_double_buffer_reader" \
                            or node.op().type() == "read":
                            flag = False
                        for tensor_node in node.inputs:
                            if tensor_node.is_var() and tensor_node.var(
                            ) is not None:
                                if tensor_node.var().type() == core.VarDesc.VarType.READER \
                                    or len(tensor_node.var().shape()) != 1:
                                    flag = False
                                    break
                        for tensor_node in node.outputs:
                            if tensor_node.is_var() and tensor_node.var(
                            ) is not None:
                                if tensor_node.var().type() == core.VarDesc.VarType.READER \
                                    or len(tensor_node.var().shape()) != 1:
                                    flag = False
                                    break
                        if flag:
                            frontier.append(node)
                            related_nodes.append(node)
                visited.add(_node_id(cur))
            return related_nodes

        # Amend the process meshes related to while_op
        for while_op_node, while_op_node_idx in self._while_op_nodes.values():
            sub_graph_id = while_op_node.op()._block_attr_id("sub_block")
            sub_graph = self._dist_context._serial_graph.get_sub_graph(
                sub_graph_id)
            sub_graph_nodes = list(sub_graph.all_nodes())
            while_dist_op = self._dist_context.get_dist_op_for_graph(
                while_op_node)
            while_op_dist_attr = while_dist_op.dist_attr

            # Step 1: set the process mesh of while_op to the merged process mesh of its subblock
            merged_process_mesh = while_op_dist_attr.process_mesh
            for node in sub_graph_nodes:
                if (node.is_var() and node.var() is not None) \
                    or (node.is_op() and node.op() is not None):
                    dist_attr = self._dist_context.get_dist_attr_for_graph(node)
                    merged_process_mesh = merge_process_mesh_two(
                        merged_process_mesh, dist_attr.process_mesh)
            while_op_dist_attr.process_mesh = merged_process_mesh

            # Step 2: set the related nodes of while_op to the process mesh of while_op
            # Step 2.1: Find related nodes of cond var the graph of while_op
            cond_tensor_related_nodes = []
            cond_tensor_name = while_op_node.op().input("Condition")[0]
            cond_tensor_node = None
            for node in while_op_node.inputs:
                if node.is_var() and node.var() is not None \
                    and node.var().name() == cond_tensor_name:
                    cond_tensor_node = node
                    cond_tensor_related_nodes.append(cond_tensor_node)
                    break

            cond_tensor_related_nodes.extend(
                _find_nodes_related_to_cond(cond_tensor_node))

            # Step 2.2: Find related nodes of cond var in the subgraph of while_op
            cond_tensor_node = None
            for node in reversed(sub_graph_nodes):
                if node.is_var() and node.var() is not None \
                    and node.var().name() == cond_tensor_name \
                        and len(node.outputs) == 0:
                    cond_tensor_node = node
                    break

            cond_tensor_related_nodes.extend(
                _find_nodes_related_to_cond(cond_tensor_node))
            # Step 2.3: Add the StepScops output of while_op
            stepscopes_tensor_name = while_op_node.op().output("StepScopes")[0]
            stepscopes_tensor_node = None
            for output_node in while_op_node.outputs:
                if output_node.is_var() and output_node.var() is not None \
                    and output_node.var().name() == stepscopes_tensor_name:
                    stepscopes_tensor_node = output_node
            cond_tensor_related_nodes.append(stepscopes_tensor_node)
            # Step 2.4: Set the process meshes of all nodes related to cond var to the process mesh of while op
            for node in cond_tensor_related_nodes:
                tensor_dist_attr = self._dist_context.get_dist_attr_for_graph(
                    node)
                tensor_dist_attr.process_mesh = merged_process_mesh

            # Step 3: set the process meshes of the inputs in while_op to the process meshes of the outside input nodes
            while_op_inputs_dist_attrs = while_op_dist_attr.inputs_dist_attrs
            for tensor_name, tensor_dist_attr in while_op_inputs_dist_attrs.items(
            ):
                nearest_tensor_node = _find_nearest_tensor_node_before(
                    self._dist_context.serial_ordered_nodes, while_op_node_idx,
                    tensor_name)
                nearest_tensor_dist_attr = self._dist_context.get_dist_attr_for_graph(
                    nearest_tensor_node)
                tensor_dist_attr.process_mesh = nearest_tensor_dist_attr.process_mesh

            # Step 4: set the process meshes of the outputs in while_op to the process meshes of the outside output nodes
            while_op_outputs_dist_attrs = while_op_dist_attr.outputs_dist_attrs
            for tensor_name, tensor_dist_attr in while_op_outputs_dist_attrs.items(
            ):
                nearest_tensor_node = _find_nearest_tensor_node_before(
                    self._dist_context.serial_ordered_nodes, while_op_node_idx,
                    tensor_name)
                if nearest_tensor_node is None:
                    nearest_tensor_node = _find_nearest_tensor_node_after(
                        self._dist_context.serial_ordered_nodes,
                        while_op_node_idx, tensor_name)
                nearest_tensor_dist_attr = self._dist_context.get_dist_attr_for_graph(
                    nearest_tensor_node)
                tensor_dist_attr.process_mesh = nearest_tensor_dist_attr.process_mesh

        # Amend the process meshes related to array
        for array_node_list in self._array_nodes.values():
            merged_process_mesh = None
            for array_node in array_node_list:
                dist_attr = self._dist_context.get_dist_attr_for_graph(
                    array_node)
                merged_process_mesh = merge_process_mesh_two(
                    merged_process_mesh, dist_attr.process_mesh)
            for array_node in array_node_list:
                dist_attr = self._dist_context.get_dist_attr_for_graph(
                    array_node)
                dist_attr.process_mesh = merged_process_mesh

    def _update_process_mesh(self):
        ordered_op_nodes = self._dist_context._serial_ordered_op_nodes

        # Step 1: Set the annotated process meshes from tensors to the first ops using them
        ordered_tensor_nodes = self._dist_context._serial_ordered_tensor_nodes
        for tensor_node in ordered_tensor_nodes:
            tensor_dist_attr = self._dist_context.get_tensor_dist_attr_for_graph(
                tensor_node)
            if not tensor_dist_attr.is_annotated("process_mesh"):
                continue
            first_op_node = None
            for op_node in ordered_op_nodes:
                # TODO: Need a better rule for the control flow ops.
                # For now, do not set the process mesh of while_op from its inputs
                if op_node.op().type() == "while":
                    continue
                for input_tensor_node in op_node.inputs:
                    if _node_id(tensor_node) == _node_id(input_tensor_node):
                        first_op_node = op_node
                        break
                if first_op_node is not None:
                    break
            if first_op_node is None:
                continue
            op_dist_attr = self._dist_context.get_dist_attr_for_graph(
                first_op_node)
            if op_dist_attr is not None and not op_dist_attr.is_annotated(
                    "process_mesh"):
                compatible_process_mesh = compute_compatible_process_mesh(
                    [tensor_dist_attr.process_mesh, op_dist_attr.process_mesh])
                if compatible_process_mesh is not None \
                    and op_dist_attr.process_mesh != compatible_process_mesh:
                    op_dist_attr.process_mesh = compatible_process_mesh

        # Step 2: set the process meshes of ops with the nearest op before them
        # Step 2.1: find the first op node which has the process mesh
        idx_of_first_op_node_has_process_mesh = -1
        for idx, op_node in enumerate(ordered_op_nodes):
            op_dist_attr = self._dist_context.get_dist_attr_for_graph(op_node)
            if op_dist_attr.process_mesh is not None \
                and idx_of_first_op_node_has_process_mesh == -1:
                idx_of_first_op_node_has_process_mesh = idx
                # Reuse the following method to set the related tensors for same op node
                self._update_process_mesh_by_nearest(op_node, op_node)
        # Step 2.2: set the process meshes of ops by the nearest op node after the first op node
        if idx_of_first_op_node_has_process_mesh + 1 > len(ordered_op_nodes):
            return None
        for idx, op_node in enumerate(ordered_op_nodes[
                idx_of_first_op_node_has_process_mesh + 1:]):
            original_idx = idx_of_first_op_node_has_process_mesh + +idx + 1
            nearest_op_node = ordered_op_nodes[original_idx - 1]
            nearest_op_dist_attr = self._dist_context.get_dist_attr_for_graph(
                nearest_op_node)
            op_dist_attr = self._dist_context.get_dist_attr_for_graph(op_node)
            assert nearest_op_dist_attr.process_mesh is not None
            self._update_process_mesh_by_nearest(op_node, nearest_op_node)
        # Step 2.3: set the process meshes of ops by the nearest op node before the first op node
        nearest_op_node = ordered_op_nodes[
            idx_of_first_op_node_has_process_mesh]
        for op_node in ordered_op_nodes[:idx_of_first_op_node_has_process_mesh]:
            self._update_process_mesh_by_nearest(op_node, nearest_op_node)

        # Step 3: adjust the process meshes for special ops
        self._update_process_mesh_for_specials()

    def _prepare(self):
        self._while_op_nodes = {}
        self._array_nodes = {}
        self._node_pairs_between_graphs = []
        all_nodes = self._dist_context.serial_ordered_nodes
        for idx, node in enumerate(all_nodes):
            if node.is_op():
                if node.op().type() == "while":
                    self._while_op_nodes[_node_id(node)] = (node, idx)
                if node.op().type() == "read_from_array":
                    array_var_name = node.op().input("X")[0]
                    if self._array_nodes.get(array_var_name, None) is None:
                        self._array_nodes[array_var_name] = []
                    self._array_nodes[array_var_name].append(node)
                if node.op().type() == "write_to_array":
                    array_var_name = node.op().output("Out")[0]
                    if self._array_nodes.get(array_var_name, None) is None:
                        self._array_nodes[array_var_name] = []
                    self._array_nodes[array_var_name].append(node)
                    self._array_nodes[array_var_name].append(node.outputs[0])
            if node.is_var() and node.var() is not None:
                if node.node.graph_id() != 0:
                    for before_node in reversed(all_nodes[:idx]):
                        if before_node.is_var() and before_node.var() is not None \
                            and before_node.node.graph_id() == node.node.graph_id() - 1 \
                                and before_node.var().name() == node.var().name():
                            self._node_pairs_between_graphs.append(
                                (before_node, node))
                    for after_node in all_nodes[idx + 1:]:
                        if after_node.is_var() and after_node.var() is not None \
                            and after_node.node.graph_id() == node.node.graph_id() - 1 \
                                and after_node.var().name() == node.var().name():
                            self._node_pairs_between_graphs.append(
                                (after_node, node))

    def complete_forward_annotation(self, serial_main_program):
        """ Complete annotation for the partial annotated serial_main_program.
        Arguments:
            serial_main_program: partial annotated serial_main_program.
        Returns:
            serial_main_program: completed annotated serial_main_program.
        """

        # Use the default distribted context for completeion if there is no one
        self._dist_context.serial_program = serial_main_program

        # Initialize distributed attributes for all var and op node in serial_main_program
        self._dist_context.init_dist_attr_for_program()
        # print_program_with_dist_attr(serial_main_program, self._dist_context)

        # Initialize distributed attributes for all var and op node in graph
        self._dist_context.init_dist_attr_for_graph()

        self._prepare()

        self._update_process_mesh()

        self._update_dims_mapping()

        # Copy the corresponding distributed attribute from graph to serial_main_program
        self._dist_context.copy_dist_attr_from_graph_to_program()
        self._dist_context.clear_dist_info_for_graph()

        # Do the validation check and amend some completion
        self._dist_context.amend_dist_attr_for_program()

        self._dist_context.validate_dist_attr_for_program()

        return serial_main_program

    def complete_backward_annotation(self, serial_main_program):
        """Complete the annotation of vars and ops in the backward phase for parallel program."""

        def _is_grad_var_name(name):
            if "@GRAD" in name:
                return True
            return False

        def _get_forward_varname_from_grad_varname(grad_var_name):
            assert _is_grad_var_name(
                grad_var_name), "[{}] is not a grad varnme.".format(
                    grad_var_name)
            return grad_var_name[:grad_var_name.find("@GRAD")]

        def _get_op_by_id(ops, id):
            for op in ops:
                if op.desc.id() == id:
                    return op
            return None

        first_backward_op_idx = -1
        for idx, op in enumerate(serial_main_program.global_block().ops):
            if int(op.attr('op_role')) == int(
                    int(core.op_proto_and_checker_maker.OpRole.Backward) | int(
                        core.op_proto_and_checker_maker.OpRole.Loss)):
                assert op.type == "fill_constant"
                first_backward_op_idx = idx
                break

        assert first_backward_op_idx >= 0, "No backward procedure found in this program."

        ops = list(serial_main_program.global_block().ops)
        vars = serial_main_program.global_block().vars
        dist_op_context = self._dist_context.dist_op_context

        for idx in range(first_backward_op_idx, len(ops)):

            # complete the initial grad loss op
            if idx == first_backward_op_idx:
                assert ops[idx].type == "fill_constant"
                assert len(
                    ops[idx].input_arg_names
                ) == 0, "first backward op should has only ONE output, but got [{}]".format(
                    len(ops[idx].input_arg_names))
                assert len(
                    ops[idx].output_arg_names
                ) == 1, "first backward op should has only ONE output, but got [{}]".format(
                    len(ops[idx].output_arg_names))

                grad_var = vars[ops[idx].output_arg_names[0]]
                forward_var_name = _get_forward_varname_from_grad_varname(
                    grad_var.name)
                forward_var = vars[forward_var_name]

                # TODO complete other attribte for grad var
                tensor_dist_attr = TensorDistributedAttribute()
                process_mesh = self._dist_context.get_tensor_dist_attr_for_program(
                    forward_var).process_mesh
                dims_mapping = self._dist_context.get_tensor_dist_attr_for_program(
                    forward_var).dims_mapping
                tensor_dist_attr.dims_mapping = dims_mapping
                tensor_dist_attr.process_mesh = process_mesh
                self._dist_context.set_tensor_dist_attr_for_program(
                    grad_var, tensor_dist_attr)

                op_dist_attr = OperatorDistributedAttribute()
                op_dist_attr.process_mesh = process_mesh
                op_dist_attr.set_output_dims_mapping(grad_var.name,
                                                     dims_mapping)
                self._dist_context.set_op_dist_attr_for_program(ops[idx],
                                                                op_dist_attr)
                continue

            # complete the annotation of grad op (xxx_grad op or sum op)
            # xxx_grad op will have a corresponding forward op in grad_op_id_to_op_id
            grad_op = ops[idx]
            if grad_op.desc.id() in dist_op_context.grad_op_id_to_op_id:
                # TODO support the case where one forward op corresponding to multiple xxx_grad op
                forward_op = _get_op_by_id(
                    ops[:first_backward_op_idx],
                    dist_op_context.grad_op_id_to_op_id[grad_op.desc.id()])
                assert forward_op is not None

                if grad_op.type == "concat" and forward_op.type == "split":
                    forward_op_dist_attr = self._dist_context.get_op_dist_attr_for_program(
                        forward_op)
                    output_var = vars[grad_op.desc.output('Out')[0]]
                    split_input_var_name = forward_op.input("X")[0]
                    ref_dims_mapping = forward_op_dist_attr.get_input_dims_mapping(
                        split_input_var_name)
                    ref_mesh = forward_op_dist_attr.process_mesh

                    grad_op_dist_attr = OperatorDistributedAttribute()
                    for input_name in grad_op.input_arg_names:
                        grad_op_dist_attr.set_input_dims_mapping(
                            input_name, ref_dims_mapping)

                    output_var_dist_attr = TensorDistributedAttribute()
                    output_var_dist_attr.dims_mapping = ref_dims_mapping
                    output_var_dist_attr.process_mesh = ref_mesh
                    self._dist_context.set_tensor_dist_attr_for_program(
                        output_var, output_var_dist_attr)

                    grad_op_dist_attr.set_output_dims_mapping(output_var.name,
                                                              ref_dims_mapping)
                    grad_op_dist_attr.process_mesh = ref_mesh
                    self._dist_context.set_op_dist_attr_for_program(
                        grad_op, grad_op_dist_attr)
                    continue

                # op dist attr
                forward_op_dist_attr = self._dist_context.get_op_dist_attr_for_program(
                    forward_op)
                forward_op_process_mesh = forward_op_dist_attr.process_mesh
                grad_op_dist_attr = OperatorDistributedAttribute()
                grad_op_dist_attr.process_mesh = forward_op_process_mesh

                # var
                for input_name in grad_op.input_arg_names:
                    input_var = vars[input_name]
                    ref_dims_mapping = None
                    if "@GRAD" in input_name:
                        forward_name = _get_forward_varname_from_grad_varname(
                            input_name)
                        ref_dims_mapping = forward_op_dist_attr.get_output_dims_mapping(
                            forward_name)
                    else:
                        if forward_op_dist_attr.get_input_dims_mapping(
                                input_name):
                            ref_dims_mapping = forward_op_dist_attr.get_input_dims_mapping(
                                input_name)
                        else:
                            ref_dims_mapping = forward_op_dist_attr.get_output_dims_mapping(
                                input_name)

                    assert ref_dims_mapping is not None, "[{}] 's dims mapping is NONE".format(
                        input_var.name)
                    grad_op_dist_attr.set_input_dims_mapping(input_name,
                                                             ref_dims_mapping)

                for output_name in grad_op.desc.output_names():
                    assert len(grad_op.desc.output(output_name)) in [0, 1]
                    if _is_grad_var_name(output_name):
                        input_name = _get_forward_varname_from_grad_varname(
                            output_name)
                    else:
                        assert grad_op.type in [
                            "cast", "c_identity", "c_allreduce_sum"
                        ]
                        input_name = "X"
                    assert input_name in forward_op.desc.input_names(
                    ), "var [{}] in op [{}]'s output but could not find [{}] in its forward op".format(
                        output_name, grad_op.type, input_name)
                    if len(grad_op.desc.output(output_name)) == 1:
                        # tensor dist attr
                        output_var = vars[grad_op.desc.output(output_name)[0]]
                        forward_name = _get_forward_varname_from_grad_varname(
                            output_var.name)
                        ref_dims_mapping = forward_op_dist_attr.get_input_dims_mapping(
                            forward_name)

                        output_var_dist_attr = TensorDistributedAttribute()
                        output_var_dist_attr.dims_mapping = ref_dims_mapping
                        output_var_dist_attr.process_mesh = forward_op_process_mesh
                        self._dist_context.set_tensor_dist_attr_for_program(
                            output_var, output_var_dist_attr)

                        grad_op_dist_attr.set_output_dims_mapping(
                            output_var.name, ref_dims_mapping)

                self._dist_context.set_op_dist_attr_for_program(
                    grad_op, grad_op_dist_attr)

            # only sum op for merge mutiple version grad has no a corresponding mapping in grad_op_id_to_op_id
            else:
                assert grad_op.type == "sum", "got unexpect op [{}]".format(
                    str(grad_op.type))
                assert all(map(_is_grad_var_name, grad_op.input_arg_names))
                assert len(grad_op.output_arg_names) == 1

                ref_forward_var_name = _get_forward_varname_from_grad_varname(
                    grad_op.output_arg_names[0])
                forward_var = vars[ref_forward_var_name]
                ref_forward_var_dims_mapping = self._dist_context.get_tensor_dist_attr_for_program(
                    forward_var).dims_mapping
                ref_forward_var_process_mesh = self._dist_context.get_tensor_dist_attr_for_program(
                    forward_var).process_mesh

                # output
                tensor_dist_attr = TensorDistributedAttribute()
                tensor_dist_attr.dims_mapping = ref_forward_var_dims_mapping
                tensor_dist_attr.process_mesh = ref_forward_var_process_mesh
                self._dist_context.set_tensor_dist_attr_for_program(
                    vars[grad_op.output_arg_names[0]], tensor_dist_attr)

                # op
                grad_op_dist_attr = OperatorDistributedAttribute()
                grad_op_dist_attr.process_mesh = ref_forward_var_process_mesh
                for var_name in grad_op.input_arg_names:
                    assert _get_forward_varname_from_grad_varname(
                        var_name) == ref_forward_var_name
                    grad_op_dist_attr.set_input_dims_mapping(
                        var_name, ref_forward_var_dims_mapping)

                grad_op_dist_attr.set_output_dims_mapping(
                    grad_op.output_arg_names[0], ref_forward_var_dims_mapping)
                self._dist_context.set_op_dist_attr_for_program(
                    grad_op, grad_op_dist_attr)

    def complete_update_annotation(self, serial_main_program):
        """Complete the annotation of vars and ops in the update phase for parallel program."""
        ops = list(serial_main_program.global_block().ops)
        vars = serial_main_program.global_block().vars
        learning_rate_completed = False

        for idx in range(len(ops)):

            # complete the annotation of the optimizer op.
            # TODO to add attribute for moment var
            op = ops[idx]
            if int(op.attr('op_role')) == int(OpRole.Optimize):
                if op.type == "clip_by_norm":
                    param_grad = vars[op.input("X")[0]]
                    param_grad_dist_attr = self._dist_context.get_tensor_dist_attr_for_program(
                        param_grad)
                    assert param_grad_dist_attr is not None
                    ref_process_mesh = param_grad_dist_attr.process_mesh
                    ref_dims_mapping = param_grad_dist_attr.dims_mapping

                    out = vars[op.output("Out")[0]]
                    out_dist_attr = TensorDistributedAttribute()
                    out_dist_attr.process_mesh = ref_process_mesh
                    out_dist_attr.dims_mapping = ref_dims_mapping
                    self._dist_context.set_tensor_dist_attr_for_program(
                        out, out_dist_attr)

                    op_dist_attr = OperatorDistributedAttribute()
                    op_dist_attr.process_mesh = ref_process_mesh
                    op_dist_attr.set_input_dist_attr(param_grad.name,
                                                     param_grad_dist_attr)
                    op_dist_attr.set_output_dist_attr(out.name, out_dist_attr)
                    self._dist_context.set_op_dist_attr_for_program(
                        op, op_dist_attr)

                if "Grad" in op.input_names and "Param" in ops[idx].input_names:
                    assert len(op.input(
                        "Param")) == 1, "Only support one-to-one now."
                    assert len(op.input(
                        "Grad")) == 1, "Only support one-to-one now."
                    param = vars[op.input("Param")[0]]
                    grad_var = vars[op.input("Grad")[0]]

                    param_dist_attr = self._dist_context.get_tensor_dist_attr_for_program(
                        param)
                    assert param_dist_attr is not None
                    ref_process_mesh = self._dist_context.get_tensor_dist_attr_for_program(
                        param).process_mesh
                    assert ref_process_mesh is not None
                    ref_dims_mapping = self._dist_context.get_tensor_dist_attr_for_program(
                        param).dims_mapping
                    assert ref_dims_mapping is not None
                    op_dist_attr = OperatorDistributedAttribute()
                    op_dist_attr.process_mesh = ref_process_mesh
                    op_dist_attr.set_input_dims_mapping(grad_var.name,
                                                        ref_dims_mapping)
                    op_dist_attr.set_input_dims_mapping(param.name,
                                                        ref_dims_mapping)
                    op_dist_attr.set_output_dims_mapping(param.name,
                                                         ref_dims_mapping)
                    learning_var = vars[op.input("LearningRate")[0]]
                    op_dist_attr.set_input_dims_mapping(learning_var.name, [-1])
                    op_dist_attr.set_output_dims_mapping(learning_var.name,
                                                         [-1])

                    if not learning_rate_completed:
                        learning_rate_completed = True
                        var_dist_attr = TensorDistributedAttribute()
                        var_dist_attr.process_mesh = ref_process_mesh
                        var_dist_attr.dims_mapping = [-1]
                        self._dist_context.set_tensor_dist_attr_for_program(
                            learning_var, var_dist_attr)

                    for input_name in op.desc.input_names():

                        if input_name in [
                                'Param', 'Grad', 'LearningRate', "SkipUpdate",
                                "Beta1Tensor", "Beta2Tensor", "EpsilonTensor",
                                "MasterParam"
                        ]:
                            continue

                        assert len(op.desc.input(input_name)) == 1
                        input_var = vars[op.desc.input(input_name)[0]]
                        input_var_attr = TensorDistributedAttribute()

                        if "Beta1Pow" in input_name or "Beta2Pow" in input_name:
                            input_var_attr.dims_mapping = [-1]
                            op_dist_attr.set_input_dims_mapping(input_var.name,
                                                                [-1])
                            op_dist_attr.set_output_dims_mapping(input_var.name,
                                                                 [-1])
                        else:
                            assert "Moment" in input_name
                            input_var_attr.dims_mapping = ref_dims_mapping
                            op_dist_attr.set_input_dims_mapping(
                                input_var.name, ref_dims_mapping)
                            op_dist_attr.set_output_dims_mapping(
                                input_var.name, ref_dims_mapping)

                        input_var_attr.process_mesh = ref_process_mesh
                        self._dist_context.set_tensor_dist_attr_for_program(
                            input_var, input_var_attr)

                    self._dist_context.set_op_dist_attr_for_program(
                        op, op_dist_attr)
                    continue
