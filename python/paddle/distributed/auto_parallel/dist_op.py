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
import paddle
from paddle.fluid import core
from paddle.fluid.framework import Variable
from .dist_attribute import TensorDistributedAttribute
from .dist_attribute import OperatorDistributedAttribute
from .dist_attribute import append_op_input_suffix
from .dist_attribute import append_op_output_suffix
from .dist_attribute import get_tensor_dist_attr_field_keys
from .dist_attribute import get_op_dist_attr_field_keys


class DistributedOperator:
    def __init__(self, serial_op, dist_attr=None):
        self._serial_op = serial_op
        self._serial_inputs = {}
        self._serial_outputs = {}
        self._dist_attr = None
        # Reuse the dist_attr setter to initialize _dist_attr
        self.dist_attr = dist_attr

    @property
    def serial_op(self):
        return self._serial_op

    @property
    def dist_attr(self):
        return self._dist_attr

    @dist_attr.setter
    def dist_attr(self, dist_attr):
        if self._dist_attr is None:
            self._dist_attr = OperatorDistributedAttribute()
        # Create new dist_attr related to current serial_op
        dist_attr = self._filter_dist_attr(dist_attr)
        # Append suffix to mark the inputs or outputs
        if isinstance(dist_attr, dict):
            # Copy the keys since we may add new ones
            for key in list(dist_attr.keys()):
                if isinstance(key, Variable):
                    if key.name in self._serial_op.input_arg_names:
                        dist_attr[append_op_input_suffix(key.name)] = True
                    if key.name in self._serial_op.output_arg_names:
                        dist_attr[append_op_output_suffix(key.name)] = True
        self._dist_attr.init(dist_attr)
        self._init_default_dist_attr()

    def get_serial_input(self, name):
        return self._serial_inputs.get(name, None)

    def get_serial_output(self, name):
        return self._serial_outputs.get(name, None)

    def _init_default_dist_attr(self):
        for tensor_name in self._serial_op.input_arg_names:
            if self._serial_op.type == "create_py_reader":
                tensor = None
            else:
                tensor = self._serial_op.block._var_recursive(tensor_name)
            self._serial_inputs[tensor_name] = tensor
            if tensor is None:
                tensor_shape = []
            else:
                if tensor.type == core.VarDesc.VarType.READER:
                    tensor_shape = []
                else:
                    tensor_shape = tensor.shape
            if self._dist_attr.get_input_dims_mapping(tensor_name) is None:
                tensor_dims_mapping = [-1 for _ in range(len(tensor_shape))]
                self._dist_attr.set_input_dims_mapping(tensor_name,
                                                       tensor_dims_mapping)
        for tensor_name in self._serial_op.output_arg_names:
            tensor = self._serial_op.block._var_recursive(tensor_name)
            if tensor.type == core.VarDesc.VarType.READER:
                tensor_shape = []
            else:
                tensor_shape = tensor.shape
            self._serial_outputs[tensor_name] = tensor
            if self._dist_attr.get_output_dims_mapping(tensor_name) is None:
                tensor_dims_mapping = [-1 for _ in range(len(tensor_shape))]
                self._dist_attr.set_output_dims_mapping(tensor_name,
                                                        tensor_dims_mapping)
        if self._dist_attr.impl_type is None:
            self._dist_attr.impl_type = "default"
        if self._dist_attr.impl_idx is None:
            self._dist_attr.impl_idx = -2

    def _filter_dist_attr(self, dist_attr):
        if dist_attr is None:
            return None
        new_dist_attr = None
        if isinstance(dist_attr, dict):
            new_dist_attr = {}
            for key, value in dist_attr.items():
                if isinstance(key, Variable):
                    if key.name in self._serial_op.input_arg_names \
                        or key.name in self._serial_op.output_arg_names:
                        new_dist_attr[key] = value
                else:
                    new_dist_attr[key] = value
        elif isinstance(dist_attr, OperatorDistributedAttribute):
            new_dist_attr = copy.deepcopy(dist_attr)
            new_dist_attr._inputs_dist_attrs.clear()
            new_dist_attr._outputs_dist_attrs.clear()
            for tensor_name in self._serial_op.input_arg_names:
                tensor_dist_attr = dist_attr.get_input_dist_attr(tensor_name)
                if tensor_dist_attr:
                    new_dist_attr.set_input_dist_attr(tensor_name,
                                                      tensor_dist_attr)
            for tensor_name in self._serial_op.output_arg_names:
                tensor_dist_attr = dist_attr.get_output_dist_attr(tensor_name)
                if tensor_dist_attr:
                    new_dist_attr.set_output_dist_attr(tensor_name,
                                                       tensor_dist_attr)
        else:
            assert False, "Cannot recognize the {} parameter.".format(dist_attr)
        return new_dist_attr

    def validate_dist_attr(self):
        if "read" in self.serial_op.type:
            return True
        for name in self.serial_op.input_arg_names:
            input_dist_attr = self.dist_attr.get_input_dist_attr(name)
            dims_mapping = input_dist_attr.dims_mapping
            shape = self.get_serial_input(name).shape
            if len(shape) != len(dims_mapping):
                return False
            for i in range(len(dims_mapping)):
                if dims_mapping[i] < -1 or dims_mapping[i] >= len(
                        self.dist_attr.process_mesh.topology):
                    return False
            for i in range(len(self.dist_attr.process_mesh.topology)):
                if dims_mapping.count(i) > 1:
                    return False
            if self.dist_attr.process_mesh != input_dist_attr.process_mesh:
                return False

        for name in self.serial_op.output_arg_names:
            output_dist_attr = self.dist_attr.get_output_dist_attr(name)
            dims_mapping = output_dist_attr.dims_mapping
            shape = self.get_serial_output(name).shape
            if len(shape) != len(dims_mapping):
                return False
            for i in range(len(dims_mapping)):
                if dims_mapping[i] < -1 or dims_mapping[i] >= len(
                        self.dist_attr.process_mesh.topology):
                    return False
            for i in range(len(self.dist_attr.process_mesh.topology)):
                if dims_mapping.count(i) > 1:
                    return False
            if self.dist_attr.process_mesh != output_dist_attr.process_mesh:
                return False
        return True

    def __str__(self):
        str = "{{op type: {}, op id: {}".format(self.serial_op.desc.type(),
                                                self.serial_op.desc.id())

        # str += ", {}".format(self.dist_attr)
        # return str

        if self.dist_attr.is_annotated("process_mesh"):
            annotated_str = "annotated"
        else:
            annotated_str = "non-annotated"
        str += ", process_mesh ({}): {}".format(annotated_str,
                                                self.dist_attr.process_mesh)

        for arg_name in self.serial_op.desc.input_arg_names():
            dims_mapping = self.dist_attr.get_input_dims_mapping(arg_name)
            if self.dist_attr.is_annotated_input_dims_mapping(arg_name):
                annotated_str = "annotated"
            else:
                annotated_str = "non-annotated"
            if self.get_serial_input(arg_name) is not None:
                if self.get_serial_input(arg_name).is_parameter:
                    is_parameter_str = "parameter"
                else:
                    is_parameter_str = "non-parameter"
            else:
                is_parameter_str = "non-parameter"
            str += ", {}'s dims_mapping (input, {}, {}): {}".format(
                arg_name, annotated_str, is_parameter_str, dims_mapping)

        for arg_name in self.serial_op.desc.output_arg_names():
            dims_mapping = self.dist_attr.get_output_dims_mapping(arg_name)
            if self.dist_attr.is_annotated_output_dims_mapping(arg_name):
                annotated_str = "annotated"
            else:
                annotated_str = "non-annotated"
            if self.get_serial_output(arg_name) is not None:
                if self.get_serial_output(arg_name).is_parameter:
                    is_parameter_str = "parameter"
                else:
                    is_parameter_str = "non-parameter"
            else:
                is_parameter_str = "non-parameter"
            str += ", {}'s dims_mapping (output, {}, {}): {}".format(
                arg_name, annotated_str, is_parameter_str, dims_mapping)

        str += ", pipeline stage: {}".format(None)

        str += ", dist_impl idx: {} }}".format(self.dist_attr._impl_idx)

        return str

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "_serial_op" or k == "_serial_inputs" or k == "_serial_outputs":
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result


class DistributedModule:
    def __init__(self, serial_module, dist_attr=None):
        self._serial_module = serial_module
        self._dist_attr = dist_attr

    def __call__(self, *args, **kwargs):
        from .dist_context import get_default_distributed_context
        main_prog = paddle.fluid.default_main_program()
        main_block = main_prog.global_block()
        op_size = len(main_block.ops)
        output = self._serial_module(*args, **kwargs)
        new_op_size = len(main_block.ops)
        default_dist_ctx = get_default_distributed_context()
        for idx in range(op_size, new_op_size):
            op = main_block.ops[idx]
            dist_op = DistributedOperator(op, self._dist_attr)
            dist_op.dist_attr.mark_annotated_as(self._dist_attr)
            default_dist_ctx.add_dist_op_for_program(dist_op)
        if isinstance(output, Variable):
            output = [output]
        return list(output)
