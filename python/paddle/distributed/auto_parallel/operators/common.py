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
# limitations under the License

import abc
from ..dist_attribute import OperatorDistributedAttribute

_g_distributed_operator_impl_containers = {}

_g_elementwise_ops = [
    "elementwise_add", "gelu", "dropout", "cast", "gather", "concat"
]
BACKWARD_ONLY_DIST_OPS = {'check_finite_and_unscale', 'update_loss_scaling'}


def is_elementwise_op(op_type):
    if op_type in _g_elementwise_ops:
        return True
    else:
        return False


class DistributedOperatorImplContainer:
    def __init__(self, op_type):
        self._type = op_type
        self._impls = []

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, op_type):
        self._type = op_type

    @property
    def impls(self):
        return self._impls

    def register_impl(self, dist_impl):
        assert self.type == dist_impl.type, \
            "Op type of container must be same as that of the implementation."
        impl_idx = len(self.impls)
        dist_impl.idx = impl_idx
        self._impls.append(dist_impl)

    def get_impl(self, impl_idx):
        return self._impls[impl_idx]

    def get_input_compatible_impls(self, dist_op):
        compatible_impls = []
        for impl in self.impls:
            if impl.is_input_compatible(dist_op):
                compatible_impls.append(impl)
        return compatible_impls

    def get_output_compatible_impls(self, dist_op):
        compatible_impls = []
        for impl in self.impls:
            if impl.is_output_compatible(dist_op):
                compatible_impls.append(impl)
        return compatible_impls

    def get_compatible_impls(self, dist_op):
        compatible_impls = []
        for impl in self.impls:
            if impl.is_auto_compatible(dist_op):
                compatible_impls.append(impl)
        return compatible_impls


class DistributedOperatorImpl(abc.ABC):
    def __init__(self, name):
        self._name = name
        self._type = None
        self._idx = None
        self._forward_implemented = False
        self._backward_implemented = False

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, op_type):
        self._type = op_type

    @property
    def idx(self):
        return self._idx

    @idx.setter
    def idx(self, impl_idx):
        self._idx = impl_idx

    @abc.abstractmethod
    def is_input_compatible(self, dist_op):
        raise NotImplementedError("Please Implement this method in Subclass.")

    @abc.abstractmethod
    def is_output_compatible(self, dist_op):
        raise NotImplementedError("Please Implement this method in Subclass.")

    @abc.abstractmethod
    def is_auto_compatible(self, dist_op):
        raise NotImplementedError("Please Implement this method in Subclass.")

    @staticmethod
    @abc.abstractmethod
    def forward(dist_ctx, *args, **kwargs):
        raise NotImplementedError("Please Implement this method in Subclass.")

    @staticmethod
    @abc.abstractmethod
    def backward(dist_ctx, *grad_outputs, **kwargs):
        raise NotImplementedError("Please Implement this method in Subclass.")

    def update_dims_mapping(self, dist_op):
        raise NotImplementedError("Please Implement this method in Subclass.")


def register_distributed_operator_impl_container(container):
    global _g_distributed_operator_impl_containers
    _g_distributed_operator_impl_containers[container.type] = container


def get_distributed_operator_impl_container(op_type):
    global _g_distributed_operator_impl_containers
    return _g_distributed_operator_impl_containers.get(op_type, None)


def register_distributed_operator_impl(op_type, dist_impl):
    dist_op_impl_container = get_distributed_operator_impl_container(op_type)
    if dist_op_impl_container is not None:
        dist_impl.type = op_type
        dist_op_impl_container.register_impl(dist_impl)
    else:
        assert False, "Must register distributed operator registry first."


def find_best_compatible_distributed_operator_impl(dist_op, fwd=True):
    """
    Here just return the first compatible implemention. 
    This will be improved by cost model in the future.
    """
    op_type = dist_op.serial_op.type
    dist_op_impl_container = get_distributed_operator_impl_container(op_type)
    dist_op_eltwise_impl_container = get_distributed_operator_impl_container(
        "elementwise")
    dist_op_default_impl_container = get_distributed_operator_impl_container(
        "default")
    compatible_impls = []
    if fwd:
        # First, find impls in the corresponding container
        if dist_op_impl_container:
            compatible_impls.extend(
                dist_op_impl_container.get_input_compatible_impls(dist_op))
        # Second, find impls in the elementwise container
        if dist_op_eltwise_impl_container and is_elementwise_op(op_type):
            compatible_impls.extend(
                dist_op_eltwise_impl_container.get_input_compatible_impls(
                    dist_op))
        # Third, find impls in the default container
        if dist_op_default_impl_container:
            compatible_impls.extend(
                dist_op_default_impl_container.get_input_compatible_impls(
                    dist_op))
    else:
        # First, find impls in the corresponding container
        if dist_op_impl_container:
            compatible_impls.extend(
                dist_op_impl_container.get_output_compatible_impls(dist_op))
        # Second, find impls in the elementwise container
        if dist_op_eltwise_impl_container and is_elementwise_op(op_type):
            compatible_impls.extend(
                dist_op_eltwise_impl_container.get_output_compatible_impls(
                    dist_op))
        # Third, find impls in the default container
        if dist_op_default_impl_container:
            compatible_impls.extend(
                dist_op_default_impl_container.get_output_compatible_impls(
                    dist_op))
    if compatible_impls:
        # For now, just return the first compatible impl
        best_compatible_impl = compatible_impls[0]
    else:
        best_compatible_impl = None
    return best_compatible_impl


def is_parameter_related(varname, block):
    if ".subprog_" in varname:
        varname = varname[:varname.index(".subprog_")]
    if ".cast_fp" in varname:
        varname = varname[:varname.index(".cast_fp")]
    assert block.has_var(varname)
    var = block.var(varname)
    return var.is_parameter


def infer_shape(block, src_var, src_var_dist_attr, op_input_dist_attr):
    var_shape = block.var(src_var.name).shape
    var_topoloy = src_var_dist_attr.process_mesh.topology
    var_dims_mapping = src_var_dist_attr.dims_mapping

    complete_shape = []
    for idx, shape in enumerate(var_shape):
        if var_dims_mapping[idx] == -1:
            complete_shape.append(shape)
        else:
            new_shape = shape * var_topoloy[var_dims_mapping[idx]]
            complete_shape.append(new_shape)

    exact_shape = []
    input_topology = op_input_dist_attr.process_mesh.topology
    input_dims_mapping = op_input_dist_attr.dims_mapping
    for idx, shape in enumerate(complete_shape):
        if input_dims_mapping[idx] == -1:
            exact_shape.append(shape)
        else:
            new_shape = shape // input_topology[input_dims_mapping[idx]]
            exact_shape.append(new_shape)

    return exact_shape


def set_comm_op_dist_attr_for_program(new_op, process_mesh, tensor_dist_attr,
                                      ctx):
    assert process_mesh is not None
    assert tensor_dist_attr is not None

    new_op_dist_attr = OperatorDistributedAttribute()
    new_op_dist_attr.process_mesh = process_mesh
    for input_varname in new_op.desc.input_arg_names():
        new_op_dist_attr.set_input_dist_attr(input_varname, tensor_dist_attr)
    for output_varname in new_op.desc.output_arg_names():
        new_op_dist_attr.set_output_dist_attr(output_varname, tensor_dist_attr)
    ctx.set_op_dist_attr_for_program(new_op, new_op_dist_attr)


def naive_copy_op_dist_attr_for_program(new_op, ref_op, ctx):

    ref_dist_attr = ctx.get_op_dist_attr_for_program(ref_op)
    new_op_dist_attr = OperatorDistributedAttribute()
    new_op_dist_attr.process_mesh = ref_dist_attr.process_mesh

    for input_name in ref_op.input_names:
        assert input_name in new_op.input_names
        assert len(ref_op.input(input_name)) == 1
        assert len(new_op.input(input_name)) == 1

        ref_tensor_dist_attr = ref_dist_attr.get_input_dist_attr(
            ref_op.input(input_name)[0])
        new_op_dist_attr.set_input_dist_attr(
            new_op.input(input_name)[0], ref_tensor_dist_attr)

    for output_name in ref_op.output_names:
        assert output_name in new_op.output_names
        assert len(ref_op.output(output_name)) == 1
        assert len(new_op.output(output_name)) == 1

        ref_tensor_dist_attr = ref_dist_attr.get_output_dist_attr(
            ref_op.output(output_name)[0])
        new_op_dist_attr.set_output_dist_attr(
            new_op.output(output_name)[0], ref_tensor_dist_attr)

    ctx.set_op_dist_attr_for_program(new_op, new_op_dist_attr)
