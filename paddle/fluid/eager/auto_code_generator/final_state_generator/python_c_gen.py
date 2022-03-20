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
import logging
from eager_gen import namespace, yaml_types_mapping, ReadFwdFile, ParseDispensable, IsVectorTensorType, GetForwardFunctionName, ParseYamlForward, DetermineForwardPositionMap, GetInplacedFunctionName, ParseInplaceInfo

###########################
## Global Configurations ##
###########################
skipped_forward_api_names = set(["scale"])


def SkipAPIGeneration(forward_api_name):
    return (forward_api_name in skipped_forward_api_names)


atype_to_parsing_function = {
    "bool": "CastPyArg2Boolean",
    "int": "CastPyArg2Int",
    "long": "CastPyArg2Long",
    "int64_t": "CastPyArg2Long",
    "float": "CastPyArg2Float",
    "std::string": "CastPyArg2String",
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


def FindParsingFunctionFromAttributeType(atype):
    if atype not in atype_to_parsing_function.keys():
        assert False, f"Unable to find {atype} in atype_to_parsing_function."

    return atype_to_parsing_function[atype]


##########################
## Refactored Functions ##
##########################
PARSE_PYTHON_C_TENSORS_TEMPLATE = \
"    auto {} = {}(\"{}\", \"{}\", args, {}, false);\n"


PARSE_PYTHON_C_ARGS_TEMPLATE = \
"""    PyObject* {}_obj = PyTuple_GET_ITEM(args, {});\n
     {} {} = {}({}_obj, \"{}\", {});\n"""


RECORD_EVENT_TEMPLATE = \
"    paddle::platform::RecordEvent {}(\"{} {}\", paddle::platform::TracerEventType::Operator, 1);"


RETURN_INPLACE_PYOBJECT_TEMPLATE = \
"""
    ssize_t arg_id = GetIdxFromCoreOpsInfoMap(core_ops_final_state_args_info, \"final_state_{}\", \"{}\");
    ssize_t return_id = GetIdxFromCoreOpsInfoMap(core_ops_final_state_returns_info, \"final_state_{}\", \"{}\");
    return ToPyObject(out, return_id, args, arg_id);
"""


PYTHON_C_FUNCTION_TEMPLATE = \
"""
static PyObject * eager_final_state_api_{}(PyObject *self, PyObject *args, PyObject *kwargs)
{{
  {}

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
{}
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


FUNCTION_NAME_TEMPLATE = \
"{}{}{}"


PYTHON_C_FUNCTION_REG_TEMPLATE = \
"{{\"final_state_{}\", (PyCFunction)(void(*)(void)) {}eager_final_state_api_{}, METH_VARARGS | METH_KEYWORDS, \"C++ interface function for {} in dygraph.\"}}"


PYTHON_C_WRAPPER_TEMPLATE = \
"""
#pragma once

#include  "pybind11/detail/common.h"
#include  "paddle/phi/api/all.h"
#include  "paddle/phi/api/lib/dygraph_api.h"
#include  "paddle/phi/common/backend.h"
#include  "paddle/phi/common/data_type.h"
#include  "paddle/phi/common/scalar.h"
#include  "paddle/phi/common/scalar_array.h"
#include  "paddle/phi/api/include/sparse_api.h"
#include  "paddle/fluid/pybind/op_function_common.h"
#include  "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include  "paddle/fluid/pybind/exception.h"
#include  "paddle/fluid/platform/profiler/event_tracing.h"
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


CORE_OPS_INFO = \
"""
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


CORE_OPS_INFO_REGISTRY = \
"""
    {\"get_final_state_core_ops_args_info\",
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

NAMESPACE_WRAPPER_TEMPLATE = \
"""namespace {} {{
    {}
}}
"""


#######################
## Generator Classes ##
#######################
class PythonCSingleFunctionGenerator:
    def __init__(self, fwd_api_contents, namespace):
        self.fwd_api_contents = fwd_api_contents
        self.namespace = namespace

        # Raw Contents
        self.forward_api_name = ""
        self.forward_args_str = ""
        self.forward_returns_str = ""

        # Raw Data
        self.forward_attrs_list = None  #[ [attr_name, attr_type, default_value, orig_position], ...]
        self.forward_inputs_list = None  #[ [arg_name, arg_type, orig_position], ...]
        self.forward_returns_list = None  #[ [ret_name, ret_type, orig_position], ...]

        # Processed Data
        self.forward_inputs_position_map = None  #{ "name" : [type, fwd_position] }
        self.forward_outputs_position_map = None  #{ "name" : [type, fwd_position] }

        # Special Op Attributes
        self.optional_inputs = []  #[name, ...]
        self.is_forward_only = True

        # Generated Results
        self.python_c_function_str = ""
        self.python_c_function_reg_str = ""

    def CollectRawContents(self):
        fwd_api_contents = self.fwd_api_contents

        assert 'api' in fwd_api_contents.keys(
        ), "Unable to find \"api\" in fwd_api_contents keys"
        assert 'args' in fwd_api_contents.keys(
        ), "Unable to find \"args\" in fwd_api_contents keys"
        assert 'output' in fwd_api_contents.keys(
        ), "Unable to find \"output\" in fwd_api_contents keys"

        self.forward_api_name = fwd_api_contents['api']
        self.forward_args_str = fwd_api_contents['args']
        self.forward_returns_str = fwd_api_contents['output']

    def CollectIsForwardOnly(self):
        fwd_api_contents = self.fwd_api_contents
        self.is_forward_only = False if 'backward' in fwd_api_contents.keys(
        ) else True

    def CollectOptionalInputs(self):
        fwd_api_contents = self.fwd_api_contents
        if 'optional' in fwd_api_contents.keys():
            self.optional_inputs = ParseDispensable(fwd_api_contents[
                'optional'])

    def CollectForwardInOutAttr(self):
        forward_args_str = self.forward_args_str
        forward_returns_str = self.forward_returns_str

        self.forward_inputs_list, self.forward_attrs_list, self.forward_returns_list = ParseYamlForward(
            forward_args_str, forward_returns_str)

    def CollectForwardPositionMap(self):
        forward_inputs_list = self.forward_inputs_list
        forward_returns_list = self.forward_returns_list

        self.forward_inputs_position_map, self.forward_outputs_position_map = DetermineForwardPositionMap(
            forward_inputs_list, forward_returns_list)

    def GeneratePythonCFunction(self, inplace_map):
        namespace = self.namespace
        forward_api_name = GetInplacedFunctionName(
            self.forward_api_name) if inplace_map else self.forward_api_name
        forward_attrs_list = self.forward_attrs_list
        forward_inputs_position_map = self.forward_inputs_position_map
        forward_outputs_position_map = self.forward_outputs_position_map
        optional_inputs = self.optional_inputs
        is_forward_only = self.is_forward_only

        # Generate Python-C Tensors Parsing Logic
        get_eager_tensor_str = ""
        for name, (ttype, pos) in forward_inputs_position_map.items():
            is_optional = (name in optional_inputs)
            if IsVectorTensorType(ttype):
                get_eager_tensor_str += PARSE_PYTHON_C_TENSORS_TEMPLATE.format(
                    name, "GetTensorListFromArgs", forward_api_name, name, pos)
            else:
                if is_optional:
                    get_eager_tensor_str += PARSE_PYTHON_C_TENSORS_TEMPLATE.format(
                        name, "GetOptionalTensorFromArgs", forward_api_name,
                        name, pos)
                else:
                    get_eager_tensor_str += PARSE_PYTHON_C_TENSORS_TEMPLATE.format(
                        name, "GetTensorFromArgs", forward_api_name, name, pos)

        parse_attributes_str = ""

        # Generate Python-C Attributes Parsing Logic
        for name, atype, _, pos in forward_attrs_list:
            parsing_function_name = FindParsingFunctionFromAttributeType(atype)
            parse_attributes_str += PARSE_PYTHON_C_ARGS_TEMPLATE.format(
                name, pos, atype, name, parsing_function_name, name,
                forward_api_name, pos)

        # Generate Dygraph Function Call Logic
        num_args = len(forward_inputs_position_map.keys()) + len(
            forward_attrs_list)
        dygraph_function_call_list = ["" for i in range(num_args)]
        for name, (_, pos) in forward_inputs_position_map.items():
            dygraph_function_call_list[pos] = f"{name}"
        for name, _, _, pos in forward_attrs_list:
            dygraph_function_call_list[pos] = f"{name}"
        dygraph_function_call_str = ",".join(dygraph_function_call_list)

        # Generate Python-C Function Definitions
        if is_forward_only:
            fwd_function_name = FUNCTION_NAME_TEMPLATE.format(
                "paddle::experimental::", namespace, forward_api_name)
        else:
            fwd_function_name = FUNCTION_NAME_TEMPLATE.format(
                "::", namespace, GetForwardFunctionName(forward_api_name))

        if inplace_map:
            assert len(
                inplace_map
            ) == 1, f"size of inplace_map must be 1, but inplace_map of \"{fwd_api_name}\" op got {len(inplace_map)}"
            for inplace_input, inplace_output in inplace_map.items():
                return_str = RETURN_INPLACE_PYOBJECT_TEMPLATE.format(
                    forward_api_name, inplace_input, forward_api_name,
                    inplace_output)
                break
        else:
            return_str = "    return ToPyObject(out);"

        # Generate Record Event for performance profiling
        pythonc_record_event_str = RECORD_EVENT_TEMPLATE.format(
            "pythonc_record_event", forward_api_name, "pybind_imperative_func")
        self.python_c_function_str = PYTHON_C_FUNCTION_TEMPLATE.format(
            forward_api_name, pythonc_record_event_str, forward_api_name,
            get_eager_tensor_str, parse_attributes_str, fwd_function_name,
            dygraph_function_call_str, return_str)

        # Generate Python-C Function Registration
        self.python_c_function_reg_str = PYTHON_C_FUNCTION_REG_TEMPLATE.format(
            forward_api_name, namespace, forward_api_name, forward_api_name)

    def run(self, inplace_map):
        # Initialized is_forward_only
        self.CollectIsForwardOnly()

        # Initialized forward_api_name, forward_args_str, forward_returns_str
        self.CollectRawContents()
        if SkipAPIGeneration(self.forward_api_name): return False

        # Initialized optional_inputs
        self.CollectOptionalInputs()

        # Initialized forward_inputs_list, forward_returns_list, forward_attrs_list
        self.CollectForwardInOutAttr()
        logging.info(
            f"Parsed Original Forward Inputs List: \n{self.forward_inputs_list}")
        logging.info(
            f"Prased Original Forward Attrs List: \n{self.forward_attrs_list}")
        logging.info(
            f"Parsed Original Forward Returns List: \n{self.forward_returns_list}"
        )

        # Initialized forward_inputs_position_map, forward_outputs_position_map
        self.CollectForwardPositionMap()
        logging.info(
            f"Generated Forward Input Position Map: {self.forward_inputs_position_map}"
        )
        logging.info(
            f"Generated Forward Output Position Map: {self.forward_outputs_position_map}"
        )

        # Code Generation
        self.GeneratePythonCFunction(inplace_map)
        logging.info(
            f"Generated Python-C Function: {self.python_c_function_str}")
        logging.info(
            f"Generated Python-C Function Declaration: {self.python_c_function_reg_str}"
        )

        return True


class PythonCYamlGenerator:
    def __init__(self, path):
        self.yaml_path = path

        self.namespace = ""
        self.forward_api_list = []

        # Generated Result
        self.python_c_functions_reg_str = ""
        self.python_c_functions_str = ""

    def ParseYamlContents(self):
        yaml_path = self.yaml_path
        self.forward_api_list = ReadFwdFile(yaml_path)

    def GeneratePythonCFunctions(self):
        namespace = self.namespace
        forward_api_list = self.forward_api_list

        for forward_api_content in forward_api_list:
            f_generator = PythonCSingleFunctionGenerator(forward_api_content,
                                                         namespace)
            status = f_generator.run({})

            if status == True:
                self.python_c_functions_reg_str += f_generator.python_c_function_reg_str + ",\n"
                self.python_c_functions_str += f_generator.python_c_function_str + "\n"

            if 'inplace' in forward_api_content.keys():
                inplace_map = ParseInplaceInfo(forward_api_content['inplace'])

                f_generator_inplace = PythonCSingleFunctionGenerator(
                    forward_api_content, namespace)
                status = f_generator_inplace.run(inplace_map)

                if status == True:
                    self.python_c_functions_reg_str += f_generator_inplace.python_c_function_reg_str + ",\n"
                    self.python_c_functions_str += f_generator_inplace.python_c_function_str + "\n"

    def InferNameSpace(self):
        yaml_path = self.yaml_path
        if "sparse" in yaml_path:
            self.namespace = "sparse::"

    def AttachNamespace(self):
        namespace = self.namespace
        python_c_functions_str = self.python_c_functions_str

        if namespace != "":
            if namespace.endswith("::"):
                namespace = namespace[:-2]
            self.python_c_functions_str = NAMESPACE_WRAPPER_TEMPLATE.format(
                namespace, python_c_functions_str)

    def run(self):
        # Infer namespace from yaml_path
        self.InferNameSpace()

        # Read Yaml file
        self.ParseYamlContents()

        # Code Generation
        self.GeneratePythonCFunctions()

        # Wrap with namespace
        self.AttachNamespace()


############################
## Code Generation Helper ##
############################
def ParseArguments():
    parser = argparse.ArgumentParser(
        description='Eager Code Generator Args Parser')
    parser.add_argument('--api_yaml_path', type=str)
    parser.add_argument('--output_path', type=str)

    args = parser.parse_args()
    return args


def GenerateCoreOpsInfoMap():
    return CORE_OPS_INFO, CORE_OPS_INFO_REGISTRY


def GeneratePythonCWrappers(python_c_function_str, python_c_function_reg_str):

    core_ops_infos_definition, core_ops_infos_registry = GenerateCoreOpsInfoMap(
    )

    python_c_function_str += core_ops_infos_definition
    python_c_function_reg_str += core_ops_infos_registry
    python_c_function_reg_str += "\n {nullptr,nullptr,0,nullptr}"

    python_c_str = PYTHON_C_WRAPPER_TEMPLATE.format(python_c_function_str,
                                                    python_c_function_reg_str)

    return python_c_str


def GeneratePythonCFile(filepath, python_c_str):
    with open(filepath, 'a') as f:
        f.write(python_c_str)


if __name__ == "__main__":
    args = ParseArguments()
    api_yaml_paths = args.api_yaml_path.split(",")

    generated_python_c_functions = ""
    generated_python_c_registration = ""
    for i in range(len(api_yaml_paths)):
        api_yaml_path = api_yaml_paths[i]

        y_generator = PythonCYamlGenerator(api_yaml_path)
        y_generator.run()

        generated_python_c_functions += y_generator.python_c_functions_str + "\n"
        generated_python_c_registration += y_generator.python_c_functions_reg_str + "\n"

    python_c_str = GeneratePythonCWrappers(generated_python_c_functions,
                                           generated_python_c_registration)

    logging.info(f"Generated Python-C Codes: \n{python_c_str}")

    output_path = args.output_path
    for path in [output_path]:
        if os.path.exists(path):
            os.remove(path)

    GeneratePythonCFile(output_path, python_c_str)
