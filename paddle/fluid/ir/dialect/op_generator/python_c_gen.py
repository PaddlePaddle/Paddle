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
    NAMESPACE_TEMPLATE,
    OP_RESULT,
    PD_MANUAL_OP_LIST,
    VECTOR_TYPE,
    CodeGen,
)

H_FILE_TEMPLATE = """

#pragma once

#include <Python.h>

// Avoid a problem with copysign defined in pyconfig.h on Windows.
#ifdef copysign
#undef copysign
#endif

{body}

"""

API_DECLARE_TEMPLATE = """
PyObject *static_api_{name}(PyObject *self, PyObject *args, PyObject *kwargs);
"""


CPP_FILE_TEMPLATE = """

#include "paddle/fluid/pybind/static_op_function.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_api.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/fluid/pybind/op_function_common.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/enforce.h"


{body}

"""

NO_MUTABLE_ATTR_API_IMPL_TEMPLATE = """
PyObject *static_api_{api_name}(PyObject *self, PyObject *args, PyObject *kwargs) {{
    try {{
        VLOG(6) << "Add {api_name} op into program";
        VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

        // Get OpResult from args
        {inputs}

        // Parse Attributes
        {attrs}

        // Call ir static api
        auto static_api_out = paddle::dialect::{api_name}({args});

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

        // Get OpResult from args
        {inputs}

        // Parse Attributes
        {attrs}

        // Call ir static api
        paddle::dialect::{api_name}({args});

        return nullptr;
    }} catch (...) {{
        ThrowExceptionToPython(std::current_exception());
        return nullptr;
    }}
}}
"""

INPUT_TEMPLATE = """
        PyObject *{name}_obj = PyTuple_GET_ITEM(args, {index});
        auto {name} = {cast_func}({name}_obj, "{api_name}", {index});"""

NO_MUTABLE_ATTR_CAST_TEMPLATE = """
        PyObject *{name}_obj = PyTuple_GET_ITEM(args, {index});
        {type} {name} = {cast_func}({name}_obj, "{api_name}", {index});"""

MUTABLE_ATTR_API_IMPL_TEMPLATE = """
PyObject *static_api_{api_name}(PyObject *self, PyObject *args, PyObject *kwargs) {{
    try {{
        VLOG(6) << "Add {api_name} op into program";
        VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

        // Get OpResult from args
        {inputs}

        // Parse Attributes
        {attrs_py_obj}

        // Check for mutable attrs
        bool has_mutable_attr = false;
        {check_mutable_attrs}

        if (has_mutable_attr){{
            {cast_attrs_with_mutable}
            // Call ir static api
            auto static_api_out = paddle::dialect::{api_name}({args_with_mutable_attrs});
            return ToPyObject(static_api_out);
        }} else {{
            {cast_attrs_without_mutable}
            // Call ir static api
            auto static_api_out = paddle::dialect::{api_name}({args_without_mutable_attrs});
            return ToPyObject(static_api_out);
        }}
    }} catch (...) {{
        ThrowExceptionToPython(std::current_exception());
        return nullptr;
    }}
}}
"""

CHECK_MUTABLE_ATTR_TEMPLATE = """
        if (PyObject_CheckIROpResult({name}_obj)){{
            has_mutable_attr = true;
        }}"""

MUTABLE_ATTR_OBJ_TEMPLATE = """
        PyObject *{name}_obj = PyTuple_GET_ITEM(args, {index});"""

MUTABLE_ATTR_CAST_TEMPLATE = """
            {type} {name} = {cast_func}({name}_obj, "{api_name}", {index});"""


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
    "Place": "CastPyArg2Place",
    "phi::DataType": "CastPyArg2DataTypeDirectly",
}


class PythonCCodeGen(CodeGen):
    def __init__(self) -> None:
        super().__init__()

    def _gen_one_declare(self, op_name):
        return API_DECLARE_TEMPLATE.format(name=op_name)

    def _gen_h_file(self, op_info_items, namespaces, h_file_path):
        declare_str = ''
        for op_info in op_info_items:
            for op_name in op_info.op_phi_name:
                # NOTE:When infer_meta_func is None, the Build() function generated in pd_op
                # is wrong, so temporarily skip the automatic generation of these APIs
                if (
                    op_info.infer_meta_func is None
                    and op_name not in PD_MANUAL_OP_LIST
                ):
                    continue
                declare_str += self._gen_one_declare(op_name)

        body = declare_str
        for namespace in reversed(namespaces):
            body = NAMESPACE_TEMPLATE.format(namespace=namespace, body=body)
        with open(h_file_path, 'w') as f:
            f.write(H_FILE_TEMPLATE.format(body=body))

    def _gen_inputs(self, op_info, op_name):
        name_list = op_info.input_name_list
        type_list = op_info.input_type_list
        assert len(name_list) == len(type_list)
        ret = ''
        for i, (name, type) in enumerate(zip(name_list, type_list)):
            cast_func = (
                'CastPyArg2VectorOfOpResult'
                if VECTOR_TYPE in type
                else 'CastPyArg2OpResult'
            )
            ret += INPUT_TEMPLATE.format(
                name=name, index=i, cast_func=cast_func, api_name=op_name
            )
        return ret

    def _gen_attrs_without_mutable(self, op_info, op_name):
        input_size = len(op_info.input_name_list)
        name_list = op_info.attribute_name_list
        type_list = op_info.attribute_build_arg_type_list
        assert len(name_list) == len(type_list)
        ret = ''
        for i, (name, type) in enumerate(zip(name_list, type_list)):
            type = type.replace('const ', '').replace('&', '')
            cast_func = TYPE_TO_FUNC_MAP[type]
            ret += NO_MUTABLE_ATTR_CAST_TEMPLATE.format(
                name=name,
                index=input_size + i,
                type=type,
                cast_func=cast_func,
                api_name=op_name,
            )
        return ret

    def _gen_attrs_py_obj_with_mutable(self, op_info):
        input_size = len(op_info.input_name_list)
        name_list = op_info.attribute_name_list
        ret = ''
        for i, name in enumerate(name_list):
            ret += MUTABLE_ATTR_OBJ_TEMPLATE.format(
                name=name, index=input_size + i
            )
        return ret

    def _gen_check_mutable_attrs(self, op_info):
        name_list = op_info.mutable_attribute_name_list
        ret = ''
        for name in name_list:
            ret += CHECK_MUTABLE_ATTR_TEMPLATE.format(name=name)
        return ret

    def _gen_cast_attrs(self, op_info, op_name, with_mutable):
        input_size = len(op_info.input_name_list)
        attr_name_list = op_info.attribute_name_list
        attr_type_list = op_info.attribute_build_arg_type_list
        mutable_attr_name_list = op_info.mutable_attribute_name_list
        assert len(attr_name_list) == len(attr_type_list)
        ret = ''
        for i, (name, type) in enumerate(zip(attr_name_list, attr_type_list)):
            type = type.replace('const ', '').replace('&', '')
            cast_func = TYPE_TO_FUNC_MAP[type]
            if with_mutable and name in mutable_attr_name_list:
                type = OP_RESULT
                cast_func = 'CastPyArg2OpResult'
            ret += MUTABLE_ATTR_CAST_TEMPLATE.format(
                type=type,
                name=name,
                cast_func=cast_func,
                api_name=op_name,
                index=input_size + i,
            )
        return ret

    def _gen_one_impl(self, op_info, op_name):
        input_name_list = op_info.input_name_list
        output_name_list = op_info.output_name_list
        attr_name_list = op_info.attribute_name_list
        mutable_attr_name_list = op_info.mutable_attribute_name_list
        no_mutable_attr_name_list = op_info.non_mutable_attribute_name_list

        if len(output_name_list) == 0:
            ret = NO_OUTPUT_API_IMPL_TEMPLATE.format(
                api_name=op_name,
                inputs=self._gen_inputs(op_info, op_name),
                attrs=self._gen_attrs_without_mutable(op_info, op_name),
                args=', '.join(input_name_list + attr_name_list),
            )
        elif len(mutable_attr_name_list) > 0:
            ret = MUTABLE_ATTR_API_IMPL_TEMPLATE.format(
                api_name=op_name,
                inputs=self._gen_inputs(op_info, op_name),
                attrs_py_obj=self._gen_attrs_py_obj_with_mutable(op_info),
                check_mutable_attrs=self._gen_check_mutable_attrs(op_info),
                cast_attrs_with_mutable=self._gen_cast_attrs(
                    op_info, op_name, True
                ),
                args_with_mutable_attrs=', '.join(
                    input_name_list
                    + mutable_attr_name_list
                    + no_mutable_attr_name_list
                ),
                cast_attrs_without_mutable=self._gen_cast_attrs(
                    op_info, op_name, False
                ),
                args_without_mutable_attrs=', '.join(
                    input_name_list + attr_name_list
                ),
            )
        else:
            ret = NO_MUTABLE_ATTR_API_IMPL_TEMPLATE.format(
                api_name=op_name,
                inputs=self._gen_inputs(op_info, op_name),
                attrs=self._gen_attrs_without_mutable(op_info, op_name),
                args=', '.join(input_name_list + attr_name_list),
            )
        ret = re.sub(r' +\n', '', ret)
        return ret

    def _gen_cpp_file(self, op_info_items, namespaces, cpp_file_path):
        impl_str = ''
        for op_info in op_info_items:
            for op_name in op_info.op_phi_name:
                # NOTE:When infer_meta_func is None, the Build() function generated in pd_op
                # is wrong, so temporarily skip the automatic generation of these APIs
                if (
                    op_info.infer_meta_func is None
                    and op_name not in PD_MANUAL_OP_LIST
                ):
                    continue
                impl_str += self._gen_one_impl(op_info, op_name)
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
    parser.add_argument('--namespaces', type=str)
    parser.add_argument('--python_c_def_h_file', type=str)
    parser.add_argument('--python_c_def_cc_file', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = ParseArguments()
    op_yaml_files = args.op_yaml_files.split(",")
    op_compat_yaml_file = args.op_compat_yaml_file
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
