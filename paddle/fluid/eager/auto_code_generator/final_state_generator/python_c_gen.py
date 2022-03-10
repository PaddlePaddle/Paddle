# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import argparse
from eager_gen import yaml_types_mapping, ReadFwdFile, ParseDispensable, IsVectorTensorType, GetForwardFunctionName, ParseYamlForward, DetermineForwardPositionMap

skipped_fwd_api_names = set(["scale"])

atype_to_parsing_function = {
    "bool": "CastPyArg2Boolean",
    "int": "CastPyArg2Int",
    "long": "CastPyArg2Long",
    "int64_t": "CastPyArg2Long",
    "float": "CastPyArg2Float",
    "string": "CastPyArg2String",
    "std::vector<bool>": "CastPyArg2Booleans",
    "std::vector<int>": "CastPyArg2Ints",
    "std::vector<long>": "CastPyArg2Longs",
    "std::vector<int64_t>": "CastPyArg2Longs",
    "std::vector<float>": "CastPyArg2Floats",
    "std::vector<double>": "CastPyArg2Float64s",
    "std::vector<std::string>": "CastPyArg2Strings",
    "paddle::experimental::Scalar": "CastPyArg2Scalar",
    "paddle::experimental::ScalarArray": "CastPyArg2ScalarArray",
    "paddle::experimental::Backend": "CastPyArg2Backend",
    "paddle::experimental::DataType": "CastPyArg2DataType",
}


def ParseArguments():
    parser = argparse.ArgumentParser(
        description='Eager Code Generator Args Parser')
    parser.add_argument('--api_yaml_path', type=str)
    parser.add_argument('--output_path', type=str)

    args = parser.parse_args()
    return args


def FindParsingFunctionFromAttributeType(atype):
    if atype not in atype_to_parsing_function.keys():
        print(f"Unable to find {atype} in atype_to_parsing_function.")
        assert False

    return atype_to_parsing_function[atype]


def GeneratePythonCFunction(fwd_api_name, forward_inputs_position_map,
                            forward_attrs_list, forward_outputs_position_map,
                            optional_inputs, is_forward_only):
    # forward_inputs_position_map = { "name" : [type, fwd_position] }
    # forward_outputs_position_map = { "name" : [type, fwd_position] }
    # forward_attrs_list = [ [attr_name, attr_type, default_value, orig_position], ...]
    # optional_inputs = [name0, ...]

    # Get EagerTensor from args
    # Get dygraph function call args
    num_args = len(forward_inputs_position_map.keys()) + len(forward_attrs_list)
    num_input_tensors = len(forward_inputs_position_map.keys())
    dygraph_function_call_list = ["" for i in range(num_args)]
    get_eager_tensor_str = ""
    for name, (ttype, pos) in forward_inputs_position_map.items():
        is_optional = (name in optional_inputs)
        if IsVectorTensorType(ttype):
            get_eager_tensor_str += f"    auto {name} = GetTensorListFromArgs(\"{fwd_api_name}\", \"{name}\", args, {pos}, false);\n"
        else:
            if is_optional:
                get_eager_tensor_str += f"    auto {name} = GetOptionalTensorFromArgs(\"{fwd_api_name}\", \"{name}\", args, {pos}, false);\n"
            else:
                get_eager_tensor_str += f"    auto {name} = GetTensorFromArgs(\"{fwd_api_name}\", \"{name}\", args, {pos}, false);\n"
        dygraph_function_call_list[pos] = f"{name}"

    parse_attributes_str = ""
    # Get Attributes
    for name, atype, _, pos in forward_attrs_list:
        parsing_function = FindParsingFunctionFromAttributeType(atype)
        key = f"{name}"

        parse_attributes_str += f"    PyObject* {name}_obj = PyTuple_GET_ITEM(args, {pos});\n"
        parse_attributes_str += f"    {atype} {name} = {parsing_function}({name}_obj, \"{fwd_api_name}\", {pos});\n"

        dygraph_function_call_list[pos] = f"{name}"
    dygraph_function_call_str = ",".join(dygraph_function_call_list)

    PYTHON_C_FUNCTION_TEMPLATE = """
static PyObject * eager_final_state_api_{}(PyObject *self, PyObject *args, PyObject *kwargs)
{{
  PyThreadState *tstate = nullptr;
  try
  {{
    VLOG(6) << "Running Eager Final State API: {}";

    // Get EagerTensors from args
{}

    // Parse Attributes
{}

    tstate = PyEval_SaveThread();
    
    auto out = {}({});
    
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  }}
  catch(...) {{
    if (tstate) {{
      PyEval_RestoreThread(tstate);
    }}
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }}
}}

"""
    if is_forward_only:
        fwd_function_name = fwd_api_name
    else:
        fwd_function_name = GetForwardFunctionName(fwd_api_name)

    python_c_function_str = PYTHON_C_FUNCTION_TEMPLATE.format(
        fwd_api_name, fwd_api_name, get_eager_tensor_str, parse_attributes_str,
        fwd_function_name, dygraph_function_call_str)

    python_c_function_reg_str = f"{{\"final_state_{fwd_api_name}\", (PyCFunction)(void(*)(void))eager_final_state_api_{fwd_api_name}, METH_VARARGS | METH_KEYWORDS, \"C++ interface function for {fwd_api_name} in dygraph.\"}}\n"

    return python_c_function_str, python_c_function_reg_str


def GenerateCoreOpsInfoMap():
    result = """
static PyObject * eager_get_final_state_core_ops_args_info(PyObject *self) {
    PyThreadState *tstate = nullptr;
    try
    {
      return ToPyObject(core_ops_final_state_args_info);
    }
    catch(...) {
      if (tstate) {
        PyEval_RestoreThread(tstate);
      }
      ThrowExceptionToPython(std::current_exception());
      return nullptr;
    }
}

static PyObject * eager_get_final_state_core_ops_args_type_info(PyObject *self) {
    PyThreadState *tstate = nullptr;
    try
    {
      return ToPyObject(core_ops_final_state_args_type_info);
    }
    catch(...) {
      if (tstate) {
        PyEval_RestoreThread(tstate);
      }
      ThrowExceptionToPython(std::current_exception());
      return nullptr;
    }
}

static PyObject * eager_get_final_state_core_ops_returns_info(PyObject *self) {
    PyThreadState *tstate = nullptr;
    try
    {
      return ToPyObject(core_ops_final_state_returns_info);
    }
    catch(...) {
      if (tstate) {
        PyEval_RestoreThread(tstate);
      }
      ThrowExceptionToPython(std::current_exception());
      return nullptr;
    }
}
    """

    core_ops_infos_registry = """
    ,{\"get_final_state_core_ops_args_info\",
    (PyCFunction)(void(*)(void))eager_get_final_state_core_ops_args_info, METH_NOARGS,
    \"C++ interface function for eager_get_final_state_core_ops_args_info.\"},
    {\"get_final_state_core_ops_args_type_info\",
    (PyCFunction)(void(*)(void))eager_get_final_state_core_ops_args_type_info,
    METH_NOARGS,
    \"C++ interface function for eager_get_final_state_core_ops_args_type_info.\"},
    {\"get_final_state_core_ops_returns_info\",
    (PyCFunction)(void(*)(void))eager_get_final_state_core_ops_returns_info,
    METH_NOARGS, \"C++ interface function for eager_get_final_state_core_ops_returns_info.\"},
"""

    return result, core_ops_infos_registry


def GeneratePythonCWrappers(python_c_function_str, python_c_function_reg_str):

    core_ops_infos_definition, core_ops_infos_registry = GenerateCoreOpsInfoMap(
    )

    python_c_function_str += core_ops_infos_definition
    python_c_function_reg_str += core_ops_infos_registry
    python_c_function_reg_str += "\n {nullptr,nullptr,0,nullptr}"

    PYTHON_C_WRAPPER_TEMPLATE = """
#pragma once

#include  "pybind11/detail/common.h"
#include  "paddle/phi/api/all.h"
#include  "paddle/phi/common/backend.h"
#include  "paddle/phi/common/data_type.h"
#include  "paddle/phi/common/scalar.h"
#include  "paddle/phi/common/scalar_array.h"
#include  "paddle/fluid/pybind/op_function_common.h"
#include  "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include  "paddle/fluid/pybind/exception.h"
#include  <Python.h>

namespace paddle {{
namespace pybind {{

{}

static PyMethodDef EagerFinalStateMethods[] = {{
    {}
}};

}} // namespace pybind
}} // namespace paddle

"""
    python_c_str = PYTHON_C_WRAPPER_TEMPLATE.format(python_c_function_str,
                                                    python_c_function_reg_str)

    return python_c_str


def GeneratePythonCFile(filepath, python_c_str):
    with open(filepath, 'a') as f:
        f.write(python_c_str)


if __name__ == "__main__":
    args = ParseArguments()

    api_yaml_path = args.api_yaml_path
    fwd_api_list = ReadFwdFile(api_yaml_path)

    python_c_function_list = []
    python_c_function_reg_list = []
    for fwd_api in fwd_api_list:

        # We only generate Ops with grad
        is_forward_only = False
        if 'backward' not in fwd_api.keys():
            is_forward_only = True

        assert 'api' in fwd_api.keys()
        assert 'args' in fwd_api.keys()
        assert 'output' in fwd_api.keys()

        fwd_api_name = fwd_api['api']
        fwd_args_str = fwd_api['args']
        fwd_returns_str = fwd_api['output']

        if fwd_api_name in skipped_fwd_api_names:
            continue

        # Parse Dispensable Inputs
        optional_inputs = []
        if 'optional' in fwd_api.keys():
            optional_inputs = ParseDispensable(fwd_api['optional'])

        # Collect Original Forward Inputs/Outputs and then perform validation checks
        forward_inputs_list, forward_attrs_list, forward_returns_list = ParseYamlForward(
            fwd_args_str, fwd_returns_str)
        print("Parsed Original Forward Inputs List: ", forward_inputs_list)
        print("Prased Original Forward Attrs List: ", forward_attrs_list)
        print("Parsed Original Forward Returns List: ", forward_returns_list)

        forward_inputs_position_map, forward_outputs_position_map = DetermineForwardPositionMap(
            forward_inputs_list, forward_returns_list)
        print("Generated Forward Input Position Map: ",
              forward_inputs_position_map)
        print("Generated Forward Output Position Map: ",
              forward_outputs_position_map)

        python_c_function_str, python_c_function_reg_str = GeneratePythonCFunction(
            fwd_api_name, forward_inputs_position_map, forward_attrs_list,
            forward_outputs_position_map, optional_inputs, is_forward_only)
        python_c_function_list.append(python_c_function_str)
        python_c_function_reg_list.append(python_c_function_reg_str)
        print("Generated Python-C Function: ", python_c_function_str)

    python_c_functions_str = "\n".join(python_c_function_list)
    python_c_functions_reg_str = ",\n".join(python_c_function_reg_list)

    python_c_str = GeneratePythonCWrappers(python_c_functions_str,
                                           python_c_functions_reg_str)

    print("Generated Python-C Codes: ", python_c_str)

    output_path = args.output_path
    for path in [output_path]:
        if os.path.exists(path):
            os.remove(path)

    GeneratePythonCFile(output_path, python_c_str)
