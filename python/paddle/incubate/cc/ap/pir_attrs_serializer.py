# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import inspect

import paddle

from ..data_type_util import get_dtype_lower_case_name
from ..typing import DType
from .py_to_axpr_json import convert_python_stmts_to_axpr_json


class PirAttrsSerializer:

    def __init__(self, func):
        self.attributes_schema = self._get_attributes_schema(func)
        self._check_attributes_schema(self.attributes_schema)
        self.attr_name2serializer = {
            attr_name: serializer
            for attr_name, schema_item in self.attributes_schema
            for serializer in [self._get_serializer(attr_name, schema_item)]
        }

    def __call__(self, **attributes):
        print(attributes)
        attributes_names = {name for name, _ in attributes.items()}
        attr_names = {name for name, _ in self.attributes_schema}
        assert (
            attributes_names == attr_names
        ), f"expected attr_names: {attr_names}, but actual attr_names are {attributes_names}"
        py_assigns = "\n".join(
            py_stmt
            for attr_name, attr_val in attributes.items()
            for py_stmt in self.attr_name2serializer[attr_name](attr_val)
        )
        py_stmts_str = f"{py_assigns}\n{self._get_attr_map_ctor_str(self.attributes_schema)}"
        return convert_python_stmts_to_axpr_json(py_stmts_str)

    def _get_attr_map_ctor_str(self, attributes_schema):
        kwargs = ", ".join(f"{name}={name}" for name, _ in attributes_schema)
        return f"__builtin__AttrMap({kwargs})"

    def _get_attributes_schema(self, obj):
        if isinstance(obj, (list, tuple)):
            return obj
        func = obj
        assert inspect.isfunction(func) or inspect.ismethod(func)
        full_arg_spec = inspect.getfullargspec(func)
        args = (
            full_arg_spec.args[1:]
            if inspect.ismethod(func)
            else full_arg_spec.args
        )
        return [
            (arg_name, annotation)
            for arg_name in args
            for annotation in [full_arg_spec.annotations[arg_name]]
        ]

    def _check_attributes_schema(self, attributes_schema):
        for _, attr_type in attributes_schema:
            self._check_attributes_schema_item_is_valid(attr_type)

    def _check_attributes_schema_item_is_valid(self, attr_type):
        if attr_type in self._supported_basic_types():
            return
        assert isinstance(
            attr_type, list
        ), f"attribute type {attr_type} is not supported."
        assert (
            len(attr_type) == 1
        ), "only syntax like [bool], [int], [float], [str] supported."
        assert (
            attr_type[0] in self._supported_basic_types()
        ), f"supported list element types are bool/int/float/str, not include {attr_type[0]}."

    def _supported_basic_types(self):
        return (bool, int, float, str, DType)

    def _get_serializer(self, attr_name, schema_item):
        assert attr_name not in (
            "custom_op_name",
            "infer_meta_func_name",
            "infer_symbolic_func_name",
        )
        schema_item_as_key = self._get_schema_item_as_key(schema_item)
        return _get_serializer_factory[schema_item_as_key](attr_name)

    def _get_schema_item_as_key(self, schema_item):
        if schema_item in self._supported_basic_types():
            return schema_item
        assert isinstance(schema_item, list)
        return tuple(schema_item)


class PirAttributeSerializer:

    def __init__(self, attr_name):
        self.attr_name = attr_name

    def __call__(self, value):
        yield from []
        raise NotImplementedError


class BoolAttributeSerializer(PirAttributeSerializer):

    def __init__(self, attr_name):
        self.attr_name = attr_name

    def __call__(self, value):
        assert isinstance(value, bool)
        yield f"{self.attr_name} = {value}"


class IntAttributeSerializer(PirAttributeSerializer):

    def __init__(self, attr_name):
        self.attr_name = attr_name

    def __call__(self, value):
        assert isinstance(value, int)
        yield f"{self.attr_name} = {value}"


class FloatAttributeSerializer(PirAttributeSerializer):

    def __init__(self, attr_name):
        self.attr_name = attr_name

    def __call__(self, value):
        assert isinstance(value, float)
        yield f"{self.attr_name} = {value}"


class StrAttributeSerializer(PirAttributeSerializer):

    def __init__(self, attr_name):
        self.attr_name = attr_name

    def __call__(self, value):
        assert isinstance(value, str)
        yield f"{self.attr_name} = {value}"


class DTypeAttributeSerializer(PirAttributeSerializer):

    def __init__(self, attr_name):
        self.attr_name = attr_name

    def __call__(self, value):
        assert isinstance(value, paddle.dtype)
        name = get_dtype_lower_case_name(value)
        yield f"{self.attr_name} = __builtin__DataType.{name}"


class BoolArrayAttributeSerializer(PirAttributeSerializer):

    def __init__(self, attr_name):
        self.attr_name = attr_name

    def __call__(self, value):
        assert isinstance(value, list)
        for elt in value:
            assert isinstance(elt, bool)
        yield f"{self.attr_name} = {value}"


class IntArrayAttributeSerializer(PirAttributeSerializer):

    def __init__(self, attr_name):
        self.attr_name = attr_name

    def __call__(self, value):
        assert isinstance(value, list)
        for elt in value:
            assert isinstance(elt, int)
        yield f"{self.attr_name} = {value}"


class FloatArrayAttributeSerializer(PirAttributeSerializer):

    def __init__(self, attr_name):
        self.attr_name = attr_name

    def __call__(self, value):
        assert isinstance(value, list)
        for elt in value:
            assert isinstance(elt, float)
        yield f"{self.attr_name} = {value}"


class StrArrayAttributeSerializer(PirAttributeSerializer):

    def __init__(self, attr_name):
        self.attr_name = attr_name

    def __call__(self, value):
        assert isinstance(value, list)
        for elt in value:
            assert isinstance(elt, str)
        yield f"{self.attr_name} = {value}"


class DTypeArrayAttributeSerializer(PirAttributeSerializer):

    def __init__(self, attr_name):
        self.attr_name = attr_name

    def __call__(self, value):
        assert isinstance(value, list)
        for elt in value:
            assert isinstance(elt, paddle.dtype)
        value_str = ", ".join(
            f"__builtin__DataType.{name}"
            for dtype in value
            for name in [get_dtype_lower_case_name(dtype)]
        )
        yield f"{self.attr_name} = [{value_str}]"


_get_serializer_factory = {
    bool: BoolAttributeSerializer,
    int: IntAttributeSerializer,
    float: FloatAttributeSerializer,
    str: StrAttributeSerializer,
    DType: DTypeAttributeSerializer,
    (bool,): BoolArrayAttributeSerializer,
    (int,): IntArrayAttributeSerializer,
    (float,): FloatArrayAttributeSerializer,
    (str,): StrArrayAttributeSerializer,
    (DType,): DTypeArrayAttributeSerializer,
}
