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
from codegen_utils import FunctionGeneratorBase, YamlGeneratorBase
from codegen_utils import yaml_types_mapping
from codegen_utils import ReadFwdFile, IsVectorTensorType, GetForwardFunctionName
from codegen_utils import ParseYamlForward, GetInplacedFunctionName

###########################
## Global Configurations ##
###########################
skipped_forward_api_names = set([])


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
    "paddle::experimental::IntArray": "CastPyArg2IntArray",
    "paddle::Place": "CastPyArg2Place",
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
"    auto {} = {}(\"{}\", \"{}\", args, {}, {});\n"


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

    // Set Device ID
{}
    
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

FUNCTION_SET_DEVICE_TEMPLATE = \
"""
    {}
    if (paddle::platform::is_gpu_place(place)) {{
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }}
"""

FUNCTION_NAME_TEMPLATE = \
"{}{}{}"


PYTHON_C_FUNCTION_REG_TEMPLATE = \
"""
{{\"final_state_{}{}\", (PyCFunction)(void(*)(void)) {}eager_final_state_api_{}, METH_VARARGS | METH_KEYWORDS, \"C++ interface function for {} in dygraph.\"}}

"""


PYTHON_C_WRAPPER_TEMPLATE = \
"""
#pragma once

#include  "pybind11/detail/common.h"
#include  "paddle/phi/api/all.h"
#include  "paddle/phi/api/lib/dygraph_api.h"
#include  "paddle/phi/common/backend.h"
#include  "paddle/phi/common/data_type.h"
#include  "paddle/phi/common/scalar.h"
#include  "paddle/phi/common/int_array.h"
#include  "paddle/phi/api/include/sparse_api.h"
#include  "paddle/phi/api/include/strings_api.h"
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
class PythonCSingleFunctionGenerator(FunctionGeneratorBase):
    def __init__(self, forward_api_contents, namespace):
        # Members from Parent:
        #self.namespace
        #self.forward_api_contents
        #self.forward_api_name
        #self.orig_forward_inputs_list
        #self.orig_forward_attrs_list
        #self.orig_forward_returns_list
        #self.forward_inputs_position_map
        #self.forward_outputs_position_map
        #self.optional_inputs
        #self.no_need_buffers
        #self.intermediate_outputs   
        #self.inplace_map
        FunctionGeneratorBase.__init__(self, forward_api_contents, namespace)

        self.is_forward_only = True

        # Generated Results
        self.python_c_function_str = ""
        self.python_c_function_reg_str = ""

    def CollectIsForwardOnly(self):
        forward_api_contents = self.forward_api_contents
        self.is_forward_only = False if 'backward' in forward_api_contents.keys(
        ) else True

    def GeneratePythonCFunction(self):
        namespace = self.namespace
        inplace_map = self.inplace_map
        forward_api_name = self.forward_api_name
        orig_forward_attrs_list = self.orig_forward_attrs_list
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
                    name, "GetTensorListFromArgs", forward_api_name, name, pos,
                    "false")
            else:
                if is_optional:
                    get_eager_tensor_str += PARSE_PYTHON_C_TENSORS_TEMPLATE.format(
                        name, "GetOptionalTensorFromArgs", forward_api_name,
                        name, pos, "true")
                else:
                    get_eager_tensor_str += PARSE_PYTHON_C_TENSORS_TEMPLATE.format(
                        name, "GetTensorFromArgs", forward_api_name, name, pos,
                        "false")

        parse_attributes_str = ""
        expected_place_str = "auto place = egr::Controller::Instance().GetExpectedPlace();\n"

        # Generate Python-C Attributes Parsing Logic
        for name, atype, _, pos in orig_forward_attrs_list:
            parsing_function_name = FindParsingFunctionFromAttributeType(atype)
            # Used input argument place if specified from Python frontend.
            if len(expected_place_str
                   ) != 0 and parsing_function_name == "CastPyArg2Place":
                expected_place_str = ""
                assert name == "place", "Only support 'place' as template argument name in FUNCTION_SET_DEVICE_TEMPLATE."

            parse_attributes_str += PARSE_PYTHON_C_ARGS_TEMPLATE.format(
                name, pos, atype, name, parsing_function_name, name,
                forward_api_name, pos)

        set_device_str = FUNCTION_SET_DEVICE_TEMPLATE.format(expected_place_str)

        # Generate Dygraph Function Call Logic
        num_args = len(forward_inputs_position_map.keys()) + len(
            orig_forward_attrs_list)
        dygraph_function_call_list = ["" for i in range(num_args)]
        for name, (_, pos) in forward_inputs_position_map.items():
            dygraph_function_call_list[pos] = f"{name}"
        for name, _, _, pos in orig_forward_attrs_list:
            dygraph_function_call_list[pos] = f"{name}"
        dygraph_function_call_str = ",".join(dygraph_function_call_list)

        # Generate Python-C Function Definitions 
        if is_forward_only:
            fwd_function_name = FUNCTION_NAME_TEMPLATE.format(
                "paddle::experimental::", namespace, forward_api_name)
        else:
            fwd_function_name = FUNCTION_NAME_TEMPLATE.format(
                "::", namespace, GetForwardFunctionName(forward_api_name))

        return_str = "    return ToPyObject(out);"

        # Generate Record Event for performance profiling
        pythonc_record_event_str = RECORD_EVENT_TEMPLATE.format(
            "pythonc_record_event", forward_api_name, "pybind_imperative_func")
        self.python_c_function_str = PYTHON_C_FUNCTION_TEMPLATE.format(
            forward_api_name, pythonc_record_event_str, forward_api_name,
            get_eager_tensor_str, parse_attributes_str, set_device_str,
            fwd_function_name, dygraph_function_call_str, return_str)

        # Set prefix of forward_api_name to avoid conflicts
        prefix = self.namespace.strip("::")
        forward_api_name_prefix = "" if prefix == "" else prefix + "_"
        # Generate Python-C Function Registration
        self.python_c_function_reg_str = PYTHON_C_FUNCTION_REG_TEMPLATE.format(
            forward_api_name_prefix, forward_api_name, namespace,
            forward_api_name, forward_api_name)

        if inplace_map:
            inplaced_forward_api_name = GetInplacedFunctionName(
                self.forward_api_name)
            if is_forward_only:
                inplaced_fwd_function_name = FUNCTION_NAME_TEMPLATE.format(
                    "paddle::experimental::", namespace,
                    inplaced_forward_api_name)
            else:
                inplaced_fwd_function_name = FUNCTION_NAME_TEMPLATE.format(
                    "::", namespace,
                    GetForwardFunctionName(inplaced_forward_api_name))

            assert len(
                inplace_map
            ) == 1, f"size of inplace_map must be 1, but inplace_map of \"{forward_api_name}\" op got {len(inplace_map)}"
            for inplace_input, inplace_output in inplace_map.items():
                return_str = RETURN_INPLACE_PYOBJECT_TEMPLATE.format(
                    inplaced_forward_api_name, inplace_input,
                    inplaced_forward_api_name, inplace_output)
                break

            self.python_c_function_str += PYTHON_C_FUNCTION_TEMPLATE.format(
                inplaced_forward_api_name, pythonc_record_event_str,
                inplaced_forward_api_name, get_eager_tensor_str,
                parse_attributes_str, set_device_str,
                inplaced_fwd_function_name, dygraph_function_call_str,
                return_str)

            # Generate Python-C Function Registration
            self.python_c_function_reg_str += "\n," + PYTHON_C_FUNCTION_REG_TEMPLATE.format(
                forward_api_name_prefix, inplaced_forward_api_name, namespace,
                inplaced_forward_api_name, inplaced_forward_api_name)

    def run(self):
        # Initialized is_forward_only
        self.CollectIsForwardOnly()

        # Initialized optional_inputs
        self.ParseDispensable()

        # Initialized inplace_map
        self.ParseInplaceInfo()

        # Initialized orig_forward_inputs_list, orig_forward_returns_list, orig_forward_attrs_list
        self.CollectOriginalForwardInfo()
        logging.info(
            f"Parsed Original Forward Inputs List: \n{self.orig_forward_inputs_list}"
        )
        logging.info(
            f"Prased Original Forward Attrs List: \n{self.orig_forward_attrs_list}"
        )
        logging.info(
            f"Parsed Original Forward Returns List: \n{self.orig_forward_returns_list}"
        )

        if SkipAPIGeneration(self.forward_api_name): return False

        # Initialized forward_inputs_position_map, forward_outputs_position_map
        self.DetermineForwardPositionMap(self.orig_forward_inputs_list,
                                         self.orig_forward_returns_list)
        logging.info(
            f"Generated Forward Input Position Map: {self.forward_inputs_position_map}"
        )
        logging.info(
            f"Generated Forward Output Position Map: {self.forward_outputs_position_map}"
        )

        # Code Generation
        self.GeneratePythonCFunction()
        logging.info(
            f"Generated Python-C Function: {self.python_c_function_str}")
        logging.info(
            f"Generated Python-C Function Declaration: {self.python_c_function_reg_str}"
        )

        return True


class PythonCYamlGenerator(YamlGeneratorBase):
    def __init__(self, path):
        # Parent members: 
        # self.namespace
        # self.api_yaml_path
        # self.forward_api_list
        YamlGeneratorBase.__init__(self, api_yaml_path)

        # Generated Result
        self.python_c_functions_reg_str = ""
        self.python_c_functions_str = ""

    def GeneratePythonCFunctions(self):
        namespace = self.namespace
        forward_api_list = self.forward_api_list

        for forward_api_content in forward_api_list:
            f_generator = PythonCSingleFunctionGenerator(forward_api_content,
                                                         namespace)
            status = f_generator.run()

            if status == True:
                self.python_c_functions_reg_str += f_generator.python_c_function_reg_str + ",\n"
                self.python_c_functions_str += f_generator.python_c_function_str + "\n"

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
        self.ParseForwardYamlContents()

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
