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
from paddle.fluid.framework import Variable
from .process_mesh import ProcessMesh

_g_tensor_dist_attr_field_keys = [
    "process_mesh", "dims_mapping", "shard_sizes", "device_placement"
]

_g_op_dist_attr_field_keys = ["process_mesh", "impl_type", "impl_idx"]

_g_op_input_suffix = "@input"

_g_op_output_suffix = "@output"


def get_tensor_dist_attr_field_keys():
    global _g_tensor_dist_attr_field_keys
    return _g_tensor_dist_attr_field_keys


def get_op_dist_attr_field_keys():
    global _g_op_dist_attr_field_keys
    return _g_op_dist_attr_field_keys


def append_op_input_suffix(name):
    global _g_op_input_suffix
    return name + _g_op_input_suffix


def append_op_output_suffix(name):
    global _g_op_output_suffix
    return name + _g_op_output_suffix


class TensorDistributedAttribute:
    def __init__(self):
        # The process mesh of distributed operator attribute must is the same as 
        # the process mesh of all input and output distributed attributed
        self._process_mesh = None
        self._dims_mapping = None
        self._shard_sizes = None
        self._device_placement = None
        self._is_annotated = {}

    @property
    def process_mesh(self):
        return self._process_mesh

    @process_mesh.setter
    def process_mesh(self, process_mesh):
        if process_mesh is not None:
            assert isinstance(process_mesh, (list, ProcessMesh)), \
                "The type of process_mesh must be list or ProcessMesh."
            if isinstance(process_mesh, list):
                process_mesh = ProcessMesh(process_mesh)
            self._process_mesh = copy.deepcopy(process_mesh)

    @property
    def dims_mapping(self):
        return self._dims_mapping

    @dims_mapping.setter
    def dims_mapping(self, dims_mapping):
        if dims_mapping is not None:
            assert isinstance(dims_mapping, list), \
                "The type of dims_mapping must be list."
            assert all(isinstance(x, int) for x in dims_mapping), \
                ("All elements of dims_mapping must be integer")
            assert all(x >= -1 for x in dims_mapping), \
                ("All elements of dims_mapping must be greater than or equal to -1.")
            self._dims_mapping = copy.deepcopy(dims_mapping)

    @property
    def shard_sizes(self):
        return self._shard_sizes

    @shard_sizes.setter
    def shard_sizes(self, shard_sizes):
        if shard_sizes is not None:
            self._shard_sizes = copy.deepcopy(shard_sizes)

    @property
    def device_placement(self):
        return self._device_placement

    @device_placement.setter
    def device_placement(self, device_placement):
        if device_placement is not None:
            self._device_placement = copy.deepcopy(device_placement)

    def init(self, dist_attr):
        if dist_attr is None:
            return
        assert isinstance(dist_attr, (dict, TensorDistributedAttribute)), \
            "The type of dist_attr must be dict or TensorDistributedAttribute."
        if dist_attr is not None:
            if isinstance(dist_attr, dict):
                for key, value in dist_attr.items():
                    if key in get_tensor_dist_attr_field_keys():
                        field_property = TensorDistributedAttribute.__dict__.get(
                            key, None)
                        if field_property:
                            field_property.fset(self, value)
                        else:
                            assert False, "No setter for {} in args {}.".format(
                                key, dist_attr)
            elif isinstance(dist_attr, TensorDistributedAttribute):
                for key in get_tensor_dist_attr_field_keys():
                    field_property = TensorDistributedAttribute.__dict__.get(
                        key, None)
                    if field_property:
                        field_property.fset(self,
                                            field_property.fget(dist_attr))
                    else:
                        assert False, "No setter for {} in args {}.".format(
                            key, dist_attr)
                self._is_annotated = copy.deepcopy(dist_attr._is_annotated)
            else:
                assert False, "Cannot recognize the {} parameter.".format(
                    dist_attr)

    def is_annotated(self, dist_attr_field_name):
        return self._is_annotated.get(dist_attr_field_name, False)

    def mark_as_annotated(self, dist_attr_field_name):
        self._is_annotated[dist_attr_field_name] = True

    def __str__(self):
        str = "\n\ttensor_dist_attr = {"
        if self.is_annotated("process_mesh"):
            annotated_str = "annotated"
        else:
            annotated_str = "non-annotated"
        str += "\n\t\tprocess_mesh ({}): {},".format(annotated_str,
                                                     self.process_mesh)

        if self.is_annotated("dims_mapping"):
            annotated_str = "annotated"
        else:
            annotated_str = "non-annotated"
        str += "\n\t\tdims_mapping ({}): {}".format(annotated_str,
                                                    self.dims_mapping)
        str += "\n\t}"
        return str


class OperatorDistributedAttribute:
    def __init__(self):
        self._process_mesh = None
        self._impl_type = None
        self._impl_idx = None
        self._inputs_dist_attrs = {}
        self._outputs_dist_attrs = {}
        self._is_annotated = {}

    @property
    def process_mesh(self):
        return self._process_mesh

    @process_mesh.setter
    def process_mesh(self, process_mesh):
        if process_mesh is not None:
            assert isinstance(process_mesh, (list, ProcessMesh)), \
                "The type of process_mesh must be list or ProcessMesh."
            if isinstance(process_mesh, list):
                process_mesh = ProcessMesh(process_mesh)
            self._process_mesh = copy.deepcopy(process_mesh)
            for dist_attr in self._inputs_dist_attrs.values():
                dist_attr.process_mesh = process_mesh
            for dist_attr in self._outputs_dist_attrs.values():
                dist_attr.process_mesh = process_mesh

    @property
    def impl_type(self):
        return self._impl_type

    @impl_type.setter
    def impl_type(self, impl_type):
        if impl_type is not None:
            self._impl_type = impl_type

    @property
    def impl_idx(self):
        return self._impl_idx

    @impl_idx.setter
    def impl_idx(self, impl_idx):
        if impl_idx is not None:
            self._impl_idx = impl_idx

    @property
    def inputs_dist_attrs(self):
        return self._inputs_dist_attrs

    @property
    def outputs_dist_attrs(self):
        return self._outputs_dist_attrs

    def get_input_dist_attr(self, name):
        return self._inputs_dist_attrs.get(name, None)

    def set_input_dist_attr(self, name, dist_attr):
        dist_attr_object = TensorDistributedAttribute()
        dist_attr_object.init(dist_attr)
        self._inputs_dist_attrs[name] = dist_attr_object

    def get_output_dist_attr(self, name):
        return self._outputs_dist_attrs.get(name, None)

    def set_output_dist_attr(self, name, dist_attr):
        dist_attr_object = TensorDistributedAttribute()
        dist_attr_object.init(dist_attr)
        self._outputs_dist_attrs[name] = dist_attr_object

    def get_input_dims_mapping(self, name):
        input_dist_attr = self.get_input_dist_attr(name)
        if input_dist_attr:
            dims_mapping = input_dist_attr.dims_mapping
        else:
            dims_mapping = None
        return dims_mapping

    def set_input_dims_mapping(self, name, dims_mapping):
        input_dist_attr = self.get_input_dist_attr(name)
        if input_dist_attr:
            input_dist_attr.dims_mapping = dims_mapping
        else:
            dist_attr = TensorDistributedAttribute()
            dist_attr.dims_mapping = dims_mapping
            self._inputs_dist_attrs[name] = dist_attr

    def get_output_dims_mapping(self, name):
        output_dist_attr = self.get_output_dist_attr(name)
        if output_dist_attr:
            dims_mapping = output_dist_attr.dims_mapping
        else:
            dims_mapping = None
        return dims_mapping

    def set_output_dims_mapping(self, name, dims_mapping):
        output_dist_attr = self.get_output_dist_attr(name)
        if output_dist_attr:
            output_dist_attr.dims_mapping = dims_mapping
        else:
            dist_attr = TensorDistributedAttribute()
            dist_attr.dims_mapping = dims_mapping
            self._outputs_dist_attrs[name] = dist_attr

    def init(self, dist_attr):
        if dist_attr is None:
            return
        assert isinstance(dist_attr, (dict, OperatorDistributedAttribute)), \
            "The type of dist_attr must be dict or OperatorDistributedAttribute."
        if dist_attr is not None:
            if isinstance(dist_attr, dict):
                for key, value in dist_attr.items():
                    if isinstance(key, Variable):
                        tensor_dist_attr = TensorDistributedAttribute()
                        tensor_dist_attr.init(value)
                        if dist_attr.get(
                                append_op_input_suffix(key.name), False):
                            self.set_input_dist_attr(key.name, tensor_dist_attr)
                        if dist_attr.get(
                                append_op_output_suffix(key.name), False):
                            self.set_output_dist_attr(key.name,
                                                      tensor_dist_attr)
                    else:
                        if key in get_op_dist_attr_field_keys():
                            field_property = OperatorDistributedAttribute.__dict__.get(
                                key, None)
                            if field_property:
                                field_property.fset(self, value)
                            else:
                                assert False, "No setter for {} in args {}.".format(
                                    key, dist_attr)
            elif isinstance(dist_attr, OperatorDistributedAttribute):
                for key in get_op_dist_attr_field_keys():
                    field_property = OperatorDistributedAttribute.__dict__.get(
                        key, None)
                    if field_property:
                        field_property.fset(self,
                                            field_property.fget(dist_attr))
                    else:
                        assert False, "No setter for {} in args {}.".format(
                            key, dist_attr)
                for tensor_name, tensor_dist_attr in dist_attr.inputs_dist_attrs.items(
                ):
                    self.set_input_dist_attr(
                        tensor_name, dist_attr.get_input_dist_attr(tensor_name))
                for tensor_name, tensor_dist_attr in dist_attr.outputs_dist_attrs.items(
                ):
                    self.set_output_dist_attr(
                        tensor_name,
                        dist_attr.get_output_dist_attr(tensor_name))
                self._is_annotated = copy.deepcopy(dist_attr._is_annotated)
            else:
                assert False, "Cannot recognize the {} parameter.".format(
                    dist_attr)

    def is_annotated(self, attr_name):
        return self._is_annotated.get(attr_name, False)

    def mark_as_annotated(self, attr_name):
        self._is_annotated[attr_name] = True

    def is_annotated_input_dims_mapping(self, name):
        input_dist_attr = self.get_input_dist_attr(name)
        if input_dist_attr:
            return input_dist_attr.is_annotated("dims_mapping")
        else:
            return False

    def is_annotated_output_dims_mapping(self, name):
        output_dist_attr = self.get_output_dist_attr(name)
        if output_dist_attr:
            return output_dist_attr.is_annotated("dims_mapping")
        else:
            return False

    def __str__(self):
        str = "\n\top_dist_attr = {"
        if self.is_annotated("process_mesh"):
            annotated_str = "annotated"
        else:
            annotated_str = "non-annotated"
        str += "\n\t\tprocess_mesh ({}): {},".format(annotated_str,
                                                     self.process_mesh)

        for arg_name in self.inputs_dist_attrs.keys():
            dims_mapping = self.get_input_dims_mapping(arg_name)
            if self.is_annotated_input_dims_mapping(arg_name):
                annotated_str = "annotated"
            else:
                annotated_str = "non-annotated"
            str += "\n\t\t{}'s dims_mapping ({}): {},".format(
                arg_name, annotated_str, dims_mapping)

        for arg_name in self.outputs_dist_attrs.keys():
            dims_mapping = self.get_output_dims_mapping(arg_name)
            if self.is_annotated_output_dims_mapping(arg_name):
                annotated_str = "annotated"
            else:
                annotated_str = "non-annotated"
            str += "\n\t\t{}'s dims_mapping ({}): {},".format(
                arg_name, annotated_str, dims_mapping)

        str += "\n\t\timpl type: {}, ".format(self._impl_type)
        str += "impl idx: {}".format(self._impl_idx)
        str += "\n\t}"
        return str
