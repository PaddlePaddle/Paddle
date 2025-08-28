# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import re

from api_gen import (
    INTARRAY_ATTRIBUTE,
    NAMESPACE_TEMPLATE,
    OP_INPUT,
    VECTOR_TYPE,
    CodeGen,
)
from gen_utils import ParsePythonAPIInfoFromYAML

args_default_mapping = {
    "x": ["input"],
    "y": ["other"],
    "axis": ["dim"],
    "keepdims": ["keepdim"],
}
# The python api info which not in ops.yaml
python_api_info_from_yaml = {}
DISABLE_TIPS = (
    "// This part of the function will be performed by a custom args mapper"
)

H_FILE_TEMPLATE = """

#pragma once

#include <Python.h>

{body}

"""

API_DECLARE_TEMPLATE = """
PyObject *static_api_{name}(PyObject *self, PyObject *args, PyObject *kwargs);
"""


CPP_FILE_TEMPLATE = """

#include "paddle/fluid/pybind/static_op_function.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/fluid/pybind/op_function_common.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/fluid/pybind/op_callstack_utils.h"
#include "paddle/fluid/pybind/arg_pre_process.h"
#include "paddle/fluid/pybind/args_mapper.h"
{body}

"""

NO_MUTABLE_ATTR_API_IMPL_TEMPLATE = """
PyObject *static_api_{api_name}(PyObject *self, PyObject *args, PyObject *kwargs) {{
    try {{
        VLOG(6) << "Add {api_name} op into program";
        VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);
        // Get Total Params count and check validity if needed
        {check_params_count}
        // Get Value from args
        {inputs}

        // Parse Attributes
        {attrs}

        // Parse input_out if needed
        {input_out}

        // Check Reminding Params validity if needed
        {check_remaining_params_valid}
        // Custom Args Mapper if need
        {custom_args_mapper}
        // Call Pre_Process before calling dygraph function if needed
        {pre_process}
        // Call ir static api
        CallStackRecorder callstack_recorder("{api_name}");
        callstack_recorder.Record();
        auto static_api_out = paddle::dialect::{api_name}({args});
        callstack_recorder.AttachToOps();
        return ToPyObject(static_api_out);
    }} catch (...) {{
        ThrowExceptionToPython(std::current_exception());
        return nullptr;
    }}
}}
"""

NO_OUTPUT_API_IMPL_TEMPLATE = """
PyObject *static_api_{api_name}(PyObject *self, PyObject *args, PyObject *kwargs) {{
    try {{
        VLOG(6) << "Add {api_name} op into program";
        VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);
        // Get Total Params count and check validity if needed
        {check_params_count}

        // Get Value from args
        {inputs}

        // Parse Attributes
        {attrs}

        // Check Reminding Params validity if needed
        {check_remaining_params_valid}
        // Custom Args Mapper if need
        {custom_args_mapper}
        // Call Pre_Process before calling dygraph function if needed
        {pre_process}

        // Call ir static api
        CallStackRecorder callstack_recorder("{api_name}");
        callstack_recorder.Record();
        paddle::dialect::{api_name}({args});
        callstack_recorder.AttachToOps();
        Py_RETURN_NONE;
    }} catch (...) {{
        ThrowExceptionToPython(std::current_exception());
        return nullptr;
    }}
}}
"""

CHECK_PARAMS_COUNT_TEMPLATE = """    int nargs = args ? static_cast<int>(PyTuple_Size(args)) : 0;
    int remaining_kwargs = kwargs ? static_cast<int>(PyDict_Size(kwargs)) : 0;
    const int max_args = {max_args};
    CheckParamsCount(nargs,remaining_kwargs,max_args);
"""
CHECK_REMAINING_PARAMS_VALID_TEMPLATE = """            CheckRemainingParamsValidity(args,kwargs,remaining_kwargs,nargs);
"""
INPUT_TEMPLATE = """
        PyObject *{name}_obj = PyTuple_GET_ITEM(args, {index});
        auto {name} = {cast_func}({name}_obj, "{api_name}", {index}, {dispensable});"""


INPUT_FROM_ARGS_KWARGS_TEMPLATE = """
        PyObject *{name}_obj = GetItemFromArgsOrKWArgs(args, {index},kwargs,{keywords}, nargs, &remaining_kwargs);
        auto {name} = {cast_func}({name}_obj, "{api_name}", {index}, {dispensable});"""

CALL_PRE_PROCESS_TEMPLATE = """{pre_process};"""
CALL_ARGS_MAPPER_TEMPLATE = """    {func_name}(args,kwargs{params});
"""
PARAMS_DECLARE_TEMPLE = """    {type} {name};\n"""
NO_MUTABLE_ATTR_CAST_TEMPLATE = """
        PyObject *{name}_obj = PyTuple_GET_ITEM(args, {index});
        {type} {name} = {cast_func}({name}_obj, "{api_name}", {index});"""

NO_MUTABLE_ATTR_CAST_FROM_ARGS_KWARGS_TEMPLATE = """
        PyObject *{name}_obj = GetItemFromArgsOrKWArgs(args, {index},kwargs,{keywords}, nargs, &remaining_kwargs,false);
        {type} {name} = {cast_func}({name}_obj, "{api_name}", {index});"""
NO_MUTABLE_ATTR_CAST_FROM_ARGS_KWARGS_WITH_DEFAULT_VALUE_TEMPLATE = """
        PyObject *{name}_obj = GetItemFromArgsOrKWArgs(args, {index},kwargs,{keywords}, nargs, &remaining_kwargs);
        {type} {name} = {cast_func}({name}_obj, "{api_name}", {index},{default_value});"""

MUTABLE_ATTR_API_IMPL_TEMPLATE = """
PyObject *static_api_{api_name}(PyObject *self, PyObject *args, PyObject *kwargs) {{
    try {{
        VLOG(6) << "Add {api_name} op into program";
        VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);
        // Get Total Params count and check validity if needed
        {check_params_count}

        // Get Value from args
        {inputs}

        // Parse Attributes
        {attrs_py_obj}

        // Parse input_out if needed
        {input_out}

        // Check for mutable attrs
        {init_attrs}
        {cast_attrs}

        // Check Reminding Params validity if needed
        {check_remaining_params_valid}
        // Custom Args Mapper if need
        {custom_args_mapper}
        // Call Pre_Process before calling dygraph function if needed
        {pre_process}

        // Call ir static api
        CallStackRecorder callstack_recorder("{api_name}");
        callstack_recorder.Record();
        auto static_api_out = paddle::dialect::{api_name}({args_with_mutable_attrs});
        callstack_recorder.AttachToOps();
        return ToPyObject(static_api_out);


    }} catch (...) {{
        ThrowExceptionToPython(std::current_exception());
        return nullptr;
    }}
}}
"""

INIT_ATTRS_TEMPLATE = """
       {type} {name};
"""
MUTABLE_ATTR_TEMPLATE = """
        if (PyObject_CheckIRValue({name}_obj)){{
            {mutable_cast_attrs}
        }}else{{
            {no_mutable_cast_attrs}
        }}"""

MUTABLE_ATTR_LIST_TEMPLATE = """
        if (PyObject_CheckIRValue({name}_obj)){{
           {mutable_cast_attrs}
        }}else if (PyObject_CheckIRVectorOfValue({name}_obj)){{
           {mutable_vector_cast_attrs}
        }}else if (PyObject_CheckIRVectorOfValueOrLong({name}_obj)){{
           {mix_vector_cast_attrs}
        }}else{{
           {no_mutable_cast_attrs}
        }}"""

MUTABLE_ATTR_OBJ_TEMPLATE = """
        PyObject *{name}_obj = PyTuple_GET_ITEM(args, {index});"""

MUTABLE_ATTR_OBJ_FROM_ARGS_KWARGS_WITH_DEFAULT_VALUE_TEMPLATE = """
        PyObject *{name}_obj = GetItemFromArgsOrKWArgs(args, {index},kwargs,{keywords}, nargs, &remaining_kwargs);"""
MUTABLE_ATTR_OBJ_FROM_ARGS_KWARGS_TEMPLATE = """
        PyObject *{name}_obj = GetItemFromArgsOrKWArgs(args, {index},kwargs,{keywords}, nargs, &remaining_kwargs,false);"""

MUTABLE_ATTR_CAST_TEMPLATE = """
            {type} {name_} = {cast_func}({name}_obj, "{api_name}", {index});"""
MUTABLE_ATTR_CAST_WITH_DEFAULT_VALUE_TEMPLATE = """
            {type} {name_} = {cast_func}({name}_obj, "{api_name}", {index}, {default_value});"""
FULL_OP_TEMPLATE = """
            {name} = paddle::dialect::full(std::vector<int64_t>{{1}}, {name}_tmp, phi::DataType::{phi_datatype}, phi::CPUPlace());
"""

FULL_INT_ARRAY_OP_TEMPLATE = """
            {name} = paddle::dialect::full_int_array({name}_tmp, phi::DataType::{phi_datatype}, phi::CPUPlace());
"""

BUILTIN_STACK_OP_TEMPLATE = """
            {name} = paddle::dialect::stack({name}_tmp, /*axis*/0);
"""
TYPE_TO_FUNC_MAP = {
    "bool": "CastPyArg2Boolean",
    "int": "CastPyArg2Int",
    "long": "CastPyArg2Long",
    "int64_t": "CastPyArg2Long",
    "float": "CastPyArg2Float",
    "double": "CastPyArg2Double",
    "std::string": "CastPyArg2String",
    "std::vector<bool>": "CastPyArg2Booleans",
    "std::vector<int>": "CastPyArg2Ints",
    "std::vector<long>": "CastPyArg2Longs",
    "std::vector<int64_t>": "CastPyArg2Longs",
    "std::vector<float>": "CastPyArg2Floats",
    "std::vector<double>": "CastPyArg2Float64s",
    "std::vector<std::string>": "CastPyArg2Strings",
    "paddle::experimental::Scalar": "CastPyArg2Scalar",
    "std::vector<phi::Scalar>": "CastPyArg2ScalarArray",
    "paddle::experimental::IntArray": "CastPyArg2IntArray",
    "paddle::Place": "CastPyArg2Place",
    "phi::Place": "CastPyArg2Place",
    "Place": "CastPyArg2Place",
    "phi::DataType": "CastPyArg2DataType",
}

TYPE_TO_PHI_DATATYPE_MAP = {
    "bool": "BOOL",
    "int": "INT32",
    "long": "INT64",
    "int64_t": "INT64",
    "float": "FLOAT32",
    "double": "FLOAT64",
    "std::vector<bool>": "BOOL",
    "std::vector<int>": "INT32",
    "std::vector<long>": "INT64",
    "std::vector<int64_t>": "INT64",
    "std::vector<float>": "FLOAT32",
    "std::vector<double>": "FLOAT64",
}

MANUAL_STATIC_OP_FUNCTION_LIST = ['full']


class PythonCCodeGen(CodeGen):
    def __init__(self) -> None:
        super().__init__()
        self.need_parse_python_api_args = False

    def _gen_one_declare(self, op_name):
        return API_DECLARE_TEMPLATE.format(name=op_name)

    def _gen_h_file(self, op_info_items, namespaces, h_file_path):
        declare_str = ''
        for op_info in op_info_items:
            for op_name in op_info.op_phi_name:
                # NOTE:When infer_meta_func is None, the Build() function generated in pd_op
                # is wrong, so temporarily skip the automatic generation of these APIs
                if self._need_skip(op_info, op_name):
                    continue
                sparse_op_name_suffix = "_sp" if op_info.is_sparse_op else ''
                sparse_op_inplace_name_suffix = (
                    "sp_" if op_info.is_sparse_op else ''
                )
                if op_name[-1] == "_":
                    declare_str += self._gen_one_declare(
                        op_name + sparse_op_inplace_name_suffix
                    )
                else:
                    declare_str += self._gen_one_declare(
                        op_name + sparse_op_name_suffix
                    )

        body = declare_str
        for namespace in reversed(namespaces):
            body = NAMESPACE_TEMPLATE.format(namespace=namespace, body=body)
        with open(h_file_path, 'w') as f:
            f.write(H_FILE_TEMPLATE.format(body=body))

    def _gen_keywords_vector(self, args_alias_map, arg_name):
        alias_set = set()
        if arg_name in args_alias_map.keys():
            alias_set = set(args_alias_map[arg_name])
        elif (
            "use_default_mapping" in args_alias_map.keys()
            and args_alias_map['use_default_mapping']
        ):
            # try to use default mapping
            if arg_name in args_default_mapping.keys():
                alias_set = set(args_default_mapping[arg_name])
        # Add the original argument name to the alias set
        alias_set.add(arg_name)
        # Convert to C++ vector format
        alias_vector = "{" + ",".join(f'"{name}"' for name in alias_set) + "}"
        return alias_vector

    def _gen_inputs(self, op_info, op_name, args_alias_map={}):
        if self.use_custom_args_mapper:
            return DISABLE_TIPS
        name_list = op_info.input_name_list
        type_list = op_info.input_type_list
        optional_list = op_info.input_optional_list
        assert len(name_list) == len(type_list) == len(optional_list)
        ret = ''
        for i, (name, type, optional) in enumerate(
            zip(name_list, type_list, optional_list)
        ):
            if optional == 'true':
                cast_func = (
                    'CastPyArg2OptionalVectorOfValue'
                    if VECTOR_TYPE in type
                    else 'CastPyArg2OptionalValue'
                )
                dispensable = "true"
            else:
                cast_func = (
                    'CastPyArg2VectorOfValue'
                    if VECTOR_TYPE in type
                    else 'CastPyArg2Value'
                )
                dispensable = "false"
            if self.need_parse_python_api_args:
                keywords = self._gen_keywords_vector(args_alias_map, name)
                ret += INPUT_FROM_ARGS_KWARGS_TEMPLATE.format(
                    name=name,
                    index=i,
                    keywords=keywords,
                    cast_func=cast_func,
                    api_name=op_name,
                    dispensable=dispensable,
                )
            else:
                ret += INPUT_TEMPLATE.format(
                    name=name,
                    index=i,
                    cast_func=cast_func,
                    api_name=op_name,
                    dispensable=dispensable,
                )
        return ret

    def _gen_attrs_without_mutable(self, op_info, op_name, args_alias_map={}):
        if self.use_custom_args_mapper:
            return DISABLE_TIPS
        input_size = len(op_info.input_name_list)
        name_list = op_info.attribute_name_list
        type_list = op_info.attribute_build_arg_type_list
        default_value_list = op_info.attribute_default_value_list
        assert len(name_list) == len(type_list)
        ret = ''
        for i, (name, type, default_value) in enumerate(
            zip(name_list, type_list, default_value_list)
        ):
            type = type.replace('const ', '').replace('&', '')
            cast_func = TYPE_TO_FUNC_MAP[type]
            if self.need_parse_python_api_args:
                keywords = self._gen_keywords_vector(args_alias_map, name)
                if default_value is not None:
                    ret += NO_MUTABLE_ATTR_CAST_FROM_ARGS_KWARGS_WITH_DEFAULT_VALUE_TEMPLATE.format(
                        name=name,
                        index=input_size + i,
                        type=type,
                        cast_func=cast_func,
                        api_name=op_name,
                        keywords=keywords,
                        default_value=default_value,
                    )
                else:
                    ret += (
                        NO_MUTABLE_ATTR_CAST_FROM_ARGS_KWARGS_TEMPLATE.format(
                            name=name,
                            index=input_size + i,
                            type=type,
                            cast_func=cast_func,
                            api_name=op_name,
                            keywords=keywords,
                        )
                    )
            else:
                ret += NO_MUTABLE_ATTR_CAST_TEMPLATE.format(
                    name=name,
                    index=input_size + i,
                    type=type,
                    cast_func=cast_func,
                    api_name=op_name,
                )
        return ret

    def _gen_attrs_py_obj_with_mutable(self, op_info, args_alias_map={}):
        if self.use_custom_args_mapper:
            return DISABLE_TIPS
        input_size = len(op_info.input_name_list)
        name_list = op_info.attribute_name_list
        default_value_list = op_info.attribute_default_value_list
        ret = ''
        for i, (name, default_value) in enumerate(
            zip(name_list, default_value_list)
        ):
            if self.need_parse_python_api_args:
                keywords = self._gen_keywords_vector(args_alias_map, name)
                if default_value is not None:
                    ret += MUTABLE_ATTR_OBJ_FROM_ARGS_KWARGS_WITH_DEFAULT_VALUE_TEMPLATE.format(
                        name=name,
                        index=input_size + i,
                        keywords=keywords,
                    )
                else:
                    ret += MUTABLE_ATTR_OBJ_FROM_ARGS_KWARGS_TEMPLATE.format(
                        name=name,
                        index=input_size + i,
                        keywords=keywords,
                    )

            else:
                ret += MUTABLE_ATTR_OBJ_TEMPLATE.format(
                    name=name, index=input_size + i
                )
        return ret

    def _gen_init_mutable_attrs(self, op_info):
        if self.use_custom_args_mapper:
            return DISABLE_TIPS
        mutable_attr_name_list = op_info.mutable_attribute_name_list
        ret = ''
        for name in mutable_attr_name_list:
            ret += INIT_ATTRS_TEMPLATE.format(type=OP_INPUT, name=name)

        return ret

    def _gen_cast_attrs(self, op_info, op_name):
        if self.use_custom_args_mapper:
            return DISABLE_TIPS
        input_size = len(op_info.input_name_list)
        attr_name_list = op_info.attribute_name_list
        attr_type_list = op_info.attribute_build_arg_type_list
        mutable_attr_name_list = op_info.mutable_attribute_name_list
        mutable_attr_type_list = op_info.mutable_attribute_type_list
        default_value_list = op_info.attribute_default_value_list
        assert len(attr_name_list) == len(attr_type_list)
        ret = ''
        for i, (name, type, default_value) in enumerate(
            zip(attr_name_list, attr_type_list, default_value_list)
        ):
            type = type.replace('const ', '').replace('&', '')
            cast_func = TYPE_TO_FUNC_MAP[type]

            if name in mutable_attr_name_list:
                phi_dtype = TYPE_TO_PHI_DATATYPE_MAP[type]
                if (
                    mutable_attr_type_list[mutable_attr_name_list.index(name)][
                        0
                    ]
                    == INTARRAY_ATTRIBUTE
                ):
                    mutable_cast_str = MUTABLE_ATTR_CAST_TEMPLATE.format(
                        type='',
                        name_=name,
                        name=name,
                        cast_func='CastPyArg2Value',
                        api_name=op_name,
                        index=input_size + i,
                    )

                    mutable_vector_cast_str = MUTABLE_ATTR_CAST_TEMPLATE.format(
                        type='std::vector<pir::Value>',
                        name_=name + '_tmp',
                        name=name,
                        cast_func='CastPyArg2VectorOfValue',
                        api_name=op_name,
                        index=input_size + i,
                    )
                    mutable_vector_cast_str += BUILTIN_STACK_OP_TEMPLATE.format(
                        name=name
                    )

                    mix_vector_cast_str = MUTABLE_ATTR_CAST_TEMPLATE.format(
                        type='std::vector<pir::Value>',
                        name_=name + '_tmp',
                        name=name,
                        cast_func='CastPyArg2VectorOfValueOrLong',
                        api_name=op_name,
                        index=input_size + i,
                    )
                    mix_vector_cast_str += BUILTIN_STACK_OP_TEMPLATE.format(
                        name=name
                    )

                else:
                    mutable_cast_str = MUTABLE_ATTR_CAST_TEMPLATE.format(
                        type='',
                        name_=name,
                        name=name,
                        cast_func='CastPyArg2Value',
                        api_name=op_name,
                        index=input_size + i,
                    )
                if default_value is not None:
                    no_mutable_cast_str = (
                        MUTABLE_ATTR_CAST_WITH_DEFAULT_VALUE_TEMPLATE.format(
                            type=type,
                            name_=name + '_tmp',
                            name=name,
                            cast_func=cast_func,
                            api_name=op_name,
                            index=input_size + i,
                            default_value=default_value,
                        )
                    )
                else:
                    no_mutable_cast_str = MUTABLE_ATTR_CAST_TEMPLATE.format(
                        type=type,
                        name_=name + '_tmp',
                        name=name,
                        cast_func=cast_func,
                        api_name=op_name,
                        index=input_size + i,
                    )

                if (
                    mutable_attr_type_list[mutable_attr_name_list.index(name)][
                        0
                    ]
                    == INTARRAY_ATTRIBUTE
                ):
                    no_mutable_cast_str += FULL_INT_ARRAY_OP_TEMPLATE.format(
                        name=name,
                        phi_datatype=phi_dtype,
                    )
                    ret += MUTABLE_ATTR_LIST_TEMPLATE.format(
                        name=name,
                        mutable_cast_attrs=mutable_cast_str,
                        mutable_vector_cast_attrs=mutable_vector_cast_str,
                        mix_vector_cast_attrs=mix_vector_cast_str,
                        no_mutable_cast_attrs=no_mutable_cast_str,
                    )
                else:
                    no_mutable_cast_str += FULL_OP_TEMPLATE.format(
                        name=name,
                        phi_datatype=phi_dtype,
                    )
                    ret += MUTABLE_ATTR_TEMPLATE.format(
                        name=name,
                        mutable_cast_attrs=mutable_cast_str,
                        no_mutable_cast_attrs=no_mutable_cast_str,
                    )
            else:
                if (
                    default_value is not None
                    and self.need_parse_python_api_args
                ):
                    mutable_cast_str = (
                        MUTABLE_ATTR_CAST_WITH_DEFAULT_VALUE_TEMPLATE.format(
                            type=type,
                            name_=name,
                            name=name,
                            cast_func=cast_func,
                            api_name=op_name,
                            index=input_size + i,
                            default_value=default_value,
                        )
                    )
                else:
                    mutable_cast_str = MUTABLE_ATTR_CAST_TEMPLATE.format(
                        type=type,
                        name_=name,
                        name=name,
                        cast_func=cast_func,
                        api_name=op_name,
                        index=input_size + i,
                    )
                ret += mutable_cast_str

        return ret

    def _gen_check_params_count(self, max_args, need_check):
        if self.use_custom_args_mapper:
            return DISABLE_TIPS
        if need_check:
            return CHECK_PARAMS_COUNT_TEMPLATE.format(max_args=max_args)
        else:
            return '// NO NEED'

    def _gen_check_reminding_params(self, need_check):
        if self.use_custom_args_mapper:
            return DISABLE_TIPS
        if need_check:
            return CHECK_REMAINING_PARAMS_VALID_TEMPLATE
        return '// NO NEED'

    def _gen_custom_args_mapper(self, op_info, args_mapper):
        if not self.use_custom_args_mapper:
            return "// NO NEED"
        args_mapper_func_name = ""
        if "static_func" in args_mapper.keys():
            args_mapper_func_name = args_mapper["static_func"]
        elif "func" in args_mapper.keys():
            args_mapper_func_name = args_mapper["func"]
        input_name_list = op_info.input_name_list
        input_type_list = op_info.input_type_list
        custom_args_mapper_str = ""
        all_params_list = []

        def _trans_dtype(dtype):
            if dtype == "paddle::dialect::DenseTensorType":
                return OP_INPUT
            # remove const exp
            if dtype.startswith("const"):
                dtype = dtype.removeprefix("const")
            if dtype.endswith("&"):
                dtype = dtype.removesuffix("&")
            return dtype

        for name, type in zip(input_name_list, input_type_list):
            custom_args_mapper_str += PARAMS_DECLARE_TEMPLE.format(
                name=name, type=_trans_dtype(type)
            )
            all_params_list.append(name)
        attribute_name_list = op_info.attribute_name_list
        attribute_type_list = op_info.attribute_build_arg_type_list
        mutable_attr_name_list = op_info.mutable_attribute_name_list
        for name, type in zip(attribute_name_list, attribute_type_list):
            if name in mutable_attr_name_list:
                type = OP_INPUT
            custom_args_mapper_str += PARAMS_DECLARE_TEMPLE.format(
                name=name, type=_trans_dtype(type)
            )
            all_params_list.append(name)

        params = ',&' + ',&'.join(all_params_list)
        custom_args_mapper_str += CALL_ARGS_MAPPER_TEMPLATE.format(
            func_name=args_mapper_func_name, params=params
        )
        return custom_args_mapper_str

    def _gen_pre_process(self, pre_process):
        if self.use_custom_args_mapper:
            return DISABLE_TIPS
        pre_process_str = ""
        if pre_process is not None and self.need_parse_python_api_args:
            if "static_func" in pre_process.keys():
                pre_process_str = pre_process["static_func"]
            elif "func" in pre_process.keys():
                pre_process_str = pre_process["func"]
            if pre_process_str != "":

                def pre_process_add_ampersand(s):
                    return (
                        s.replace('(', '(&').replace(',', ',&').rstrip(')')
                        + ')'
                    )

                return CALL_PRE_PROCESS_TEMPLATE.format(
                    pre_process=pre_process_add_ampersand(pre_process_str)
                )
        return "// NO NEED"

    def _gen_one_impl(self, op_info, op_name):
        input_name_list = op_info.input_name_list
        output_name_list = op_info.output_name_list
        attr_name_list = op_info.attribute_name_list
        mutable_attr_name_list = op_info.mutable_attribute_name_list
        no_mutable_attr_name_list = op_info.non_mutable_attribute_name_list
        max_args = len(input_name_list) + len(attr_name_list)
        python_api_info = op_info.python_api_info
        args_alias_map = None
        pre_process = None
        args_mapper = None
        need_check_params_count = False
        self.need_parse_python_api_args = False
        self.use_custom_args_mapper = False
        # Do not parse sparse op's python_api_info
        if (
            not op_info.is_sparse_op
        ) and op_name in python_api_info_from_yaml.keys():
            python_api_info = python_api_info_from_yaml[op_name]
        if python_api_info is not None:
            self.need_parse_python_api_args = True
            if "args_alias" in python_api_info.keys():
                args_alias_map = python_api_info["args_alias"]
                need_check_params_count = True
            if "pre_process" in python_api_info.keys():
                pre_process = python_api_info["pre_process"]
            if "args_mapper" in python_api_info.keys():
                args_mapper = python_api_info["args_mapper"]
                if args_mapper is not None and (
                    "static_func" in args_mapper.keys()
                    or "func" in args_mapper.keys()
                ):
                    self.use_custom_args_mapper = True

        if len(output_name_list) == 0:
            ret = NO_OUTPUT_API_IMPL_TEMPLATE.format(
                api_name=op_name,
                check_params_count=self._gen_check_params_count(
                    max_args, need_check=need_check_params_count
                ),
                inputs=self._gen_inputs(op_info, op_name, args_alias_map),
                attrs=self._gen_attrs_without_mutable(
                    op_info, op_name, args_alias_map
                ),
                check_remaining_params_valid=self._gen_check_reminding_params(
                    need_check=need_check_params_count
                ),
                custom_args_mapper=self._gen_custom_args_mapper(
                    op_info=op_info, args_mapper=args_mapper
                ),
                pre_process=self._gen_pre_process(pre_process),
                args=', '.join(input_name_list + attr_name_list),
            )
        elif len(mutable_attr_name_list) > 0:
            get_input_out_str = ""
            if (
                not op_name[-1:] == "_"
                and not op_name[-4:] == "grad"
                and "sparse" not in op_name
            ):
                get_input_out_str = "Check_PIR_not_support_out(kwargs);"
            ret = MUTABLE_ATTR_API_IMPL_TEMPLATE.format(
                api_name=op_name,
                check_params_count=self._gen_check_params_count(
                    max_args, need_check=need_check_params_count
                ),
                inputs=self._gen_inputs(op_info, op_name, args_alias_map),
                attrs_py_obj=self._gen_attrs_py_obj_with_mutable(
                    op_info, args_alias_map
                ),
                init_attrs=self._gen_init_mutable_attrs(op_info),
                cast_attrs=self._gen_cast_attrs(op_info, op_name),
                check_remaining_params_valid=self._gen_check_reminding_params(
                    need_check=need_check_params_count
                ),
                custom_args_mapper=self._gen_custom_args_mapper(
                    op_info, args_mapper
                ),
                pre_process=self._gen_pre_process(pre_process),
                args_with_mutable_attrs=', '.join(
                    input_name_list
                    + mutable_attr_name_list
                    + no_mutable_attr_name_list
                ),
                input_out=get_input_out_str,
            )
        else:
            get_input_out_str = ""
            if (
                not op_name[-1:] == "_"
                and not op_name[-4:] == "grad"
                and "sparse" not in op_name
            ):
                get_input_out_str = "Check_PIR_not_support_out(kwargs);"
            ret = NO_MUTABLE_ATTR_API_IMPL_TEMPLATE.format(
                api_name=op_name,
                check_params_count=self._gen_check_params_count(
                    max_args, need_check=need_check_params_count
                ),
                inputs=self._gen_inputs(op_info, op_name, args_alias_map),
                attrs=self._gen_attrs_without_mutable(
                    op_info, op_name, args_alias_map
                ),
                custom_args_mapper=self._gen_custom_args_mapper(
                    op_info, args_mapper
                ),
                args=', '.join(input_name_list + attr_name_list),
                check_remaining_params_valid=self._gen_check_reminding_params(
                    need_check=need_check_params_count
                ),
                pre_process=self._gen_pre_process(pre_process),
                input_out=get_input_out_str,
            )
        ret = re.sub(r' +\n', '', ret)
        return ret

    def _need_skip(self, op_info, op_name):
        return (
            super()._need_skip(op_info, op_name)
            or op_name.endswith('xpu')
            or op_name in MANUAL_STATIC_OP_FUNCTION_LIST
        )

    def _gen_cpp_file(self, op_info_items, namespaces, cpp_file_path):
        impl_str = ''
        for op_info in op_info_items:
            for op_name in op_info.op_phi_name:
                # NOTE:When infer_meta_func is None, the Build() function generated in pd_op
                # is wrong, so temporarily skip the automatic generation of these APIs
                if self._need_skip(op_info, op_name):
                    continue
                sparse_op_name_suffix = "_sp" if op_info.is_sparse_op else ''
                sparse_op_inplace_name_suffix = (
                    "sp_" if op_info.is_sparse_op else ''
                )
                if op_name[-1] == "_":
                    impl_str += self._gen_one_impl(
                        op_info, op_name + sparse_op_inplace_name_suffix
                    )
                else:
                    impl_str += self._gen_one_impl(
                        op_info, op_name + sparse_op_name_suffix
                    )
        body = impl_str
        for namespace in reversed(namespaces):
            body = NAMESPACE_TEMPLATE.format(namespace=namespace, body=body)
        with open(cpp_file_path, 'w') as f:
            f.write(CPP_FILE_TEMPLATE.format(body=body))


def ParseArguments():
    parser = argparse.ArgumentParser(
        description='Generate Dialect Python C Files By Yaml'
    )
    parser.add_argument('--op_yaml_files', type=str)
    parser.add_argument('--op_compat_yaml_file', type=str)
    parser.add_argument('--python_api_info_yaml_path', type=str)
    parser.add_argument('--namespaces', type=str)
    parser.add_argument('--python_c_def_h_file', type=str)
    parser.add_argument('--python_c_def_cc_file', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = ParseArguments()
    op_yaml_files = args.op_yaml_files.split(",")
    op_compat_yaml_file = args.op_compat_yaml_file

    python_api_info_yaml_path = args.python_api_info_yaml_path
    python_api_info_from_yaml = ParsePythonAPIInfoFromYAML(
        python_api_info_yaml_path
    )

    if args.namespaces is not None:
        namespaces = args.namespaces.split(",")
    python_c_def_h_file = args.python_c_def_h_file
    python_c_def_cc_file = args.python_c_def_cc_file

    code_gen = PythonCCodeGen()
    code_gen.gen_h_and_cpp_file(
        op_yaml_files,
        op_compat_yaml_file,
        namespaces,
        python_c_def_h_file,
        python_c_def_cc_file,
    )
