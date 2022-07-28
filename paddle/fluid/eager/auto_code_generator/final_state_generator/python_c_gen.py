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
from codegen_utils import FunctionGeneratorBase, GeneratorBase
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
"""    PyObject* {}_obj = PyTuple_GET_ITEM(args, {});
    {} {} = {}({}_obj, \"{}\", {});
"""


RECORD_EVENT_TEMPLATE = \
"paddle::platform::RecordEvent {}(\"{} {}\", paddle::platform::TracerEventType::Operator, 1);"


RETURN_INPLACE_PYOBJECT_TEMPLATE = \
"""
    inplace_var_idx_map[{}] = {};
"""

LAYOUT_LOGIC_TEMPLATE=\
"""
  if (paddle::imperative::LayoutAutoTune::Instance().UseLayoutAutoTune()) {{
    VLOG(5) << "Check and Prepare For LAYOUT";
    paddle::small_vector<std::vector<paddle::experimental::Tensor>, egr::kSlotSmallVectorSize> tensors_vector = {};
    {}
    {} 
    // Call dygraph function
    decltype({}({})) out = {}({});
    {}
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    {}
}}
"""

PYTHON_C_FUNCTION_TEMPLATE = \
"""
static PyObject * eager_final_state_api_{}(PyObject *self, PyObject *args, PyObject *kwargs) {{
  {}

  PyThreadState *tstate = nullptr;
  try {{
    VLOG(6) << "Running Eager Final State API: {}";

    // Get EagerTensors from args
{}
    // Parse Attributes if needed
{}
    tstate = PyEval_SaveThread();

    // Set Device ID
{}
    // LayoutAutoTune
    {} 
    // Call dygraph function
    decltype({}({})) out = {}({});

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    {}
  }} catch(...) {{
    if (tstate) {{
      PyEval_RestoreThread(tstate);
    }}
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }}
}}
"""

FUNCTION_SET_DEVICE_TEMPLATE = \
"""{}    if (paddle::platform::is_gpu_place(place)) {{
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
#include "paddle/fluid/eager/eager_layout_auto_tune.h"
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
        #self.forward_inplace_map
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
        forward_inplace_map = self.forward_inplace_map
        forward_api_name = self.forward_api_name
        orig_forward_attrs_list = self.orig_forward_attrs_list
        forward_inputs_position_map = self.forward_inputs_position_map
        forward_outputs_position_map = self.forward_outputs_position_map
        optional_inputs = self.optional_inputs
        is_forward_only = self.is_forward_only
        intermediate_outputs = self.intermediate_outputs

        inplace_args_pos_map = {}
        inplace_returns_pos_map = {}
        # Generate Python-C Tensors Parsing Logic
        get_eager_tensor_str = ""
        # layout autotune
        layout_tensors_vector_list = []
        layout_autotune_list = []
        layout_tensors_vector_optional_list = []
        for name, (ttype, pos) in forward_inputs_position_map.items():
            if forward_inplace_map and name in forward_inplace_map.keys():
                inplace_args_pos_map[name] = pos
            is_optional = (name in optional_inputs)
            if IsVectorTensorType(ttype):
                get_eager_tensor_str += PARSE_PYTHON_C_TENSORS_TEMPLATE.format(
                    name, "GetTensorListFromArgs", forward_api_name, name, pos,
                    "false")
                #layout 1.set layout_tensors_vector
                layout_tensors_vector_list.append(f"{name}")
                #layout 3. trans in tensor
                layout_autotune_list.append(f"auto new_{name} = {name};\n")
            else:
                if is_optional:
                    get_eager_tensor_str += PARSE_PYTHON_C_TENSORS_TEMPLATE.format(
                        name, "GetOptionalTensorFromArgs", forward_api_name,
                        name, pos, "true")
                    layout_tensors_vector_optional_list.append(
                        f"if ({name}) tensors_vector.push_back({{ *{name} }});\n"
                    )
                    layout_autotune_list.append(f"auto new_{name} = {name};\n")
                else:
                    get_eager_tensor_str += PARSE_PYTHON_C_TENSORS_TEMPLATE.format(
                        name, "GetTensorFromArgs", forward_api_name, name, pos,
                        "false")
                    layout_tensors_vector_list.append(f"{{{name}}}")
                    #layout 3. trans in tensor
                    layout_autotune_list.append(
                        f"auto new_{name} = transformer->TransInTensor(\"{name}\", {name});\n"
                    )

        if forward_inplace_map:
            for name, (ttype, pos) in forward_outputs_position_map.items():
                if name in forward_inplace_map.values():
                    inplace_returns_pos_map[name] = pos

        parse_attributes_str = ""
        expected_place_str = "    auto place = egr::Controller::Instance().GetExpectedPlace();\n"

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
        num_args = len(
            forward_inputs_position_map.keys()) + len(orig_forward_attrs_list)
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

        # Autotune Layout logic
        # layout.2 get transformer
        lightly_sensitive_attr = [
            'axis',
            'axes',
            'dim',
            'dims',
            'start',
            'end',
            'stop',
            'keepdims'  # for argmax
        ]
        heavily_sensitive_attr = ['data_format', 'data_layout']
        layout_autotune_attr = []
        layout_autotune_attr_code_list = []
        layout_autotune_attr_type_list = []
        layout_autotune_attr_code_list.append(
            f"auto op_name = phi::TransToFluidOpName(\"{forward_api_name}\");\n"
        )
        for name, atype, default_val, pos in orig_forward_attrs_list:
            lightly_flag = False
            heavily_flag = False
            for attr_name in lightly_sensitive_attr:
                if name.find(
                        attr_name) != -1 and name not in layout_autotune_attr:
                    lightly_flag = True
                    layout_autotune_attr.append(name)
                    layout_autotune_attr_type_list.append(atype)
            if lightly_flag is False:
                for attr_name in heavily_sensitive_attr:
                    if name.find(attr_name
                                 ) != -1 and name not in layout_autotune_attr:
                        layout_autotune_attr.append(name)
                        layout_autotune_attr_type_list.append(atype)
                        heavily_flag = True
        if len(layout_autotune_attr) == 0:
            layout_autotune_attr_code_list.append(
                f"auto transformer = egr::EagerLayoutAutotune(op_name, tensors_vector);\n"
            )
        elif len(layout_autotune_attr) == 1:
            layout_autotune_attr_code_list.append(
                f"auto transformer = egr::EagerLayoutAutotune<{layout_autotune_attr_type_list[0]}>(op_name, tensors_vector, &{layout_autotune_attr[0]});\n"
            )
        elif len(layout_autotune_attr) == 2:
            layout_autotune_attr_code_list.append(
                f"auto transformer = egr::EagerLayoutAutotune<{layout_autotune_attr_type_list[0]}, {layout_autotune_attr_type_list[1]}>(op_name, tensors_vector, &{layout_autotune_attr[0]}, &{layout_autotune_attr[1]});\n"
            )
        else:
            layout_autotune_attr_code_list.append(
                f"auto transformer = egr::EagerLayoutAutotune(op_name, tensors_vector, {len(layout_autotune_attr)});\n"
            )
        #layout 4. call forward api
        layout_function_call_list = ["" for i in range(num_args)]
        for name, (_, pos) in forward_inputs_position_map.items():
            layout_function_call_list[pos] = f"new_{name}"
        for name, _, _, pos in orig_forward_attrs_list:
            layout_function_call_list[pos] = f"{name}"
        layout_function_call_str = ",".join(layout_function_call_list)
        #layout 5. set output tensor
        num_outputs = len(
            forward_outputs_position_map.keys()) - len(intermediate_outputs)
        returns_list = []
        for name, (rtype, pos) in forward_outputs_position_map.items():
            if name in intermediate_outputs:
                continue
            returns_list.append(f"{name}")
        layout_tmp_result_list = []
        layout_autotune_outs_list = ""
        if num_outputs == 1:
            returns_str = returns_list[0]
            layout_autotune_outs_list += f"  auto& new_out = out;\n"
            layout_autotune_outs_list += f"  transformer -> SetOutTensorLayout(&new_out);\n"
        else:
            for name, (rtype, pos) in forward_outputs_position_map.items():
                if name in intermediate_outputs:
                    continue
                layout_autotune_outs_list += f"  auto& new_{name} = std::get<{len(layout_tmp_result_list)}>(out);\n"
                layout_autotune_outs_list += f"  transformer -> SetOutTensorLayout(&new_{name});\n"
                layout_tmp_result_list.append(f"{name}")

        #layoutfinal.
        layout_tensors_vector_optional_list_str = "".join(
            layout_tensors_vector_optional_list)
        layout_tensors_vector_list_str = "{ " + ",".join(
            layout_tensors_vector_list) + " }"
        layout_logic_str = LAYOUT_LOGIC_TEMPLATE.format(
            layout_tensors_vector_list_str,
            layout_tensors_vector_optional_list_str,
            "    ".join(layout_autotune_attr_code_list) + "    " +
            "    ".join(layout_autotune_list), fwd_function_name,
            layout_function_call_str, fwd_function_name,
            layout_function_call_str, layout_autotune_outs_list, return_str)
        if is_forward_only is False:
            layout_logic_str = ""

        # Generate Python-C Function Definetion
        self.python_c_function_str = PYTHON_C_FUNCTION_TEMPLATE.format(
            forward_api_name, pythonc_record_event_str, forward_api_name,
            get_eager_tensor_str, parse_attributes_str, set_device_str,
            layout_logic_str, fwd_function_name, dygraph_function_call_str,
            fwd_function_name, dygraph_function_call_str, return_str)

        # Set prefix of forward_api_name to avoid conflicts
        prefix = self.namespace.strip("::")
        forward_api_name_prefix = "" if prefix == "" else prefix + "_"

        # Generate Python-C Function Registration
        self.python_c_function_reg_str = PYTHON_C_FUNCTION_REG_TEMPLATE.format(
            forward_api_name_prefix, forward_api_name, namespace,
            forward_api_name, forward_api_name)

        if forward_inplace_map:
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

            return_str = "    std::map<ssize_t, ssize_t> inplace_var_idx_map;"
            for inplace_input, inplace_output in forward_inplace_map.items():
                return_str += RETURN_INPLACE_PYOBJECT_TEMPLATE.format(
                    inplace_returns_pos_map[inplace_output],
                    inplace_args_pos_map[inplace_input])
            return_str += "    return ToPyObject(out, args, inplace_var_idx_map);"

            # Generate Python-C Function Definetion
            python_c_inplace_func_str = PYTHON_C_FUNCTION_TEMPLATE.format(
                inplaced_forward_api_name, pythonc_record_event_str,
                inplaced_forward_api_name, get_eager_tensor_str,
                parse_attributes_str, set_device_str, "",
                inplaced_fwd_function_name, dygraph_function_call_str,
                inplaced_fwd_function_name, dygraph_function_call_str,
                return_str)

            python_c_inplace_func_reg_str = PYTHON_C_FUNCTION_REG_TEMPLATE.format(
                forward_api_name_prefix, inplaced_forward_api_name, namespace,
                inplaced_forward_api_name, inplaced_forward_api_name)

            # self.forward_api_name ending with '_' means it only has inplace api
            if self.forward_api_name[-1] == '_':
                self.python_c_function_str = python_c_inplace_func_str
                # Generate Python-C Function Registration
                self.python_c_function_reg_str = python_c_inplace_func_reg_str
            else:
                self.python_c_function_str += python_c_inplace_func_str
                # Generate Python-C Function Registration
                self.python_c_function_reg_str += "\n," + python_c_inplace_func_reg_str

    def run(self):
        # Initialized is_forward_only
        self.CollectIsForwardOnly()

        # Initialized optional_inputs
        self.ParseDispensable()

        # Initialized forward_inplace_map
        self.ParseForwardInplaceInfo()

        # Initialized orig_forward_inputs_list, orig_forward_returns_list, orig_forward_attrs_list
        self.CollectOriginalForwardInfo()

        if SkipAPIGeneration(self.forward_api_name): return False

        # Initialized forward_inputs_position_map, forward_outputs_position_map
        self.DetermineForwardPositionMap(self.orig_forward_inputs_list,
                                         self.orig_forward_returns_list)

        # Code Generation
        self.GeneratePythonCFunction()

        return True


class PythonCGenerator(GeneratorBase):

    def __init__(self, path):
        # Parent members:
        # self.namespace
        # self.api_yaml_path
        # self.forward_api_list
        GeneratorBase.__init__(self, api_yaml_path)

        # Generated Result
        self.python_c_functions_str = ""
        self.python_c_functions_reg_str = ""

    def GeneratePythonCFunctions(self):
        namespace = self.namespace
        forward_api_list = self.forward_api_list

        for forward_api_content in forward_api_list:
            f_generator = PythonCSingleFunctionGenerator(
                forward_api_content, namespace)
            status = f_generator.run()

            if status == True:
                self.python_c_functions_str += f_generator.python_c_function_str + "\n"
                self.python_c_functions_reg_str += f_generator.python_c_function_reg_str + ",\n"

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

        py_c_generator = PythonCGenerator(api_yaml_path)
        py_c_generator.run()

        generated_python_c_functions += py_c_generator.python_c_functions_str + "\n"
        generated_python_c_registration += py_c_generator.python_c_functions_reg_str + "\n"

    python_c_str = GeneratePythonCWrappers(generated_python_c_functions,
                                           generated_python_c_registration)

    output_path = args.output_path
    for path in [output_path]:
        if os.path.exists(path):
            os.remove(path)

    GeneratePythonCFile(output_path, python_c_str)
