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

import argparse

from codegen_utils import (
    FunctionGeneratorBase,
    GeneratorBase,
    ParsePythonAPIInfoFromYAML,
)

IMPORT_TEMPLATE = """
import paddle
from paddle import _C_ops
from .. import core
"""

FUNCTION_NAME_TEMPLATE = """
def {func_name}():
"""

NAME_METHOD_MAPPING_TEMPLATE = """  ('{op_name}',_{op_name})"""

METHODS_MAP_TEMPLATE = """
methods_map = [
{}
]

"""
FUNCTIONS_MAP_TEMPLATE = """
funcs_map = [
{}
]

"""
NN_FUNCTIONS_MAP_TEMPLATE = """
nn_funcs_map = [
{}
]

"""

METHOD_TEMPLATE = """
def _{name}(*args, **kwargs):
    return _C_ops.{name}(*args, **kwargs)
"""
SET_METHOD_TEMPLATE = """
    # set methods for paddle.Tensor in dygraph
    local_tensor = core.eager.Tensor
    for method_name, method in methods_map:
        setattr(local_tensor, method_name, method)
        setattr(paddle.tensor, method_name, method)

"""
SET_FUNCTION_TEMPLATE = """
    # set functions for paddle
    for method_name, method in funcs_map:
        setattr(paddle, method_name, method)

"""
SET_NN_FUNCTION_TEMPLATE = """
    # set functions for paddle.nn.functional
    for method_name, method in nn_funcs_map:
        setattr(paddle.nn.functional, method_name, method)
"""
# The pair of name and func which should be added to paddle
paddle_func_map = []
# The pair of name and func which should be added to paddle.Tensor
tensor_method_map = []
# The pair of name and func which should be added to paddle.nn.functional
nn_func_map = []
# The python api info which not in ops.yaml
python_api_info_from_yaml = {}


class MethodGenerator(FunctionGeneratorBase):
    def __init__(self, forward_api_contents, namespace):
        FunctionGeneratorBase.__init__(self, forward_api_contents, namespace)
        self.need_parse_python_api_args = False
        # Generated Results
        self.Method_str = ""

    def run(self):
        # Initialized orig_forward_inputs_list, orig_forward_returns_list, orig_forward_attrs_list
        self.CollectOriginalForwardInfo()
        if len(self.python_api_info) > 0:
            self.need_parse_python_api_args = True
            self.ParsePythonAPIInfo()
            self.Method_str = GenerateMethod(self.forward_api_name)
            ClassifyAPIByPrefix(self.python_api_info, self.forward_api_name)


def ExtractPrefix(full_name):
    res = ""
    for m in full_name.split(".")[:-1]:
        res += m + '.'
    return res


def GenerateMethod(name):
    return METHOD_TEMPLATE.format(name=name)


def ClassifyAPIByPrefix(python_api_info, op_name):
    python_api_names = python_api_info["name"]
    name_func_mapping = NAME_METHOD_MAPPING_TEMPLATE.format(op_name=op_name)
    for name in python_api_names:
        prefix = ExtractPrefix(name)
        if prefix == "paddle.":
            paddle_func_map.append(name_func_mapping)
        elif prefix == "paddle.Tensor.":
            tensor_method_map.append(name_func_mapping)
        elif prefix == "paddle.nn.functional.":
            nn_func_map.append(name_func_mapping)
        else:
            raise Exception("Unsupported Prefix " + prefix, "API : " + name)


class MonkeyPatchTensorMethodsGenerator(GeneratorBase):
    def __init__(self, path):
        # Parent members:
        # self.namespace
        # self.api_yaml_path
        # self.forward_api_list
        GeneratorBase.__init__(self, path)

        # Generated Result
        self.MonkeyPatchTensorMethods_str = ""

    def GenerateMonkeyPatchTensorMethods(self):
        self.MonkeyPatchTensorMethods_str += IMPORT_TEMPLATE

        forward_api_list = self.forward_api_list
        methods_map = []  # [("method_name",method),]
        method_str = ""
        # some python api info in ops.yaml
        for forward_api_content in forward_api_list:
            f_generator = MethodGenerator(forward_api_content, None)
            status = f_generator.run()
            method_str += f_generator.Method_str
        # some python api info not in ops.yaml but in python_api_info.yaml
        for ops_name, python_api_info in python_api_info_from_yaml.items():
            method_str += GenerateMethod(ops_name)
            ClassifyAPIByPrefix(python_api_info, ops_name)

        self.MonkeyPatchTensorMethods_str += method_str
        result = ',\n '.join(tensor_method_map)
        self.MonkeyPatchTensorMethods_str += METHODS_MAP_TEMPLATE.format(result)
        result = ',\n '.join(paddle_func_map)
        self.MonkeyPatchTensorMethods_str += FUNCTIONS_MAP_TEMPLATE.format(
            result
        )
        result = ',\n '.join(nn_func_map)
        self.MonkeyPatchTensorMethods_str += NN_FUNCTIONS_MAP_TEMPLATE.format(
            result
        )
        self.MonkeyPatchTensorMethods_str += FUNCTION_NAME_TEMPLATE.format(
            func_name="monkey_patch_generated_methods_for_tensor"
        )
        self.MonkeyPatchTensorMethods_str += SET_METHOD_TEMPLATE
        self.MonkeyPatchTensorMethods_str += SET_FUNCTION_TEMPLATE
        self.MonkeyPatchTensorMethods_str += SET_NN_FUNCTION_TEMPLATE

    def run(self):
        # Read Yaml file
        self.ParseForwardYamlContents()
        self.GenerateMonkeyPatchTensorMethods()


##########################
# Code Generation Helper #
##########################
def ParseArguments():
    parser = argparse.ArgumentParser(
        description='Eager Code Generator Args Parser for Monkey patch methods '
    )
    parser.add_argument('--api_yaml_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--python_api_info_yaml_path', type=str)
    args = parser.parse_args()
    return args


def GenerateMonkeyPathFile(filepath, python_c_str):
    with open(filepath, 'w') as f:
        f.write(python_c_str)


if __name__ == "__main__":
    args = ParseArguments()
    api_yaml_path = args.api_yaml_path
    output_path = args.output_path
    python_api_info_yaml_path = args.python_api_info_yaml_path

    python_api_info_from_yaml = ParsePythonAPIInfoFromYAML(
        python_api_info_yaml_path
    )

    gen = MonkeyPatchTensorMethodsGenerator(api_yaml_path)
    gen.run()
    GenerateMonkeyPathFile(output_path, gen.MonkeyPatchTensorMethods_str)
