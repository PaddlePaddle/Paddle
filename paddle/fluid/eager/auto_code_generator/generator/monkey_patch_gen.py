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
)

IMPORT_TEMPLATE = """
import paddle
from paddle import _C_ops
from .. import core
"""

FUNCTION_NAME_TEMPLATE = """
def {func_name}():
"""

NAME_METHOD_MAPPING_TEMPLATE = """  ('{api_name}',_{api_name})"""

METHODS_MAP_TEMPLATE = """
methods_map = [
{}
]
"""

METHOD_TEMPLATE = """
def _{name}(self,*args, **kwargs):
    return _C_ops.{name}(self,*args, **kwargs)
"""
SET_METHOD_TEMPLATE = """
    # set methods for Tensor in dygraph
    local_tensor = core.eager.Tensor
    for method_name, method in methods_map:
        setattr(local_tensor, method_name, method)

"""


class MethodGenerator(FunctionGeneratorBase):
    def __init__(self, forward_api_contents, namespace):
        FunctionGeneratorBase.__init__(self, forward_api_contents, namespace)
        self.need_parse_python_api_args = False
        # Generated Results
        self.Method_str = ""

    def GenerateMethod(self, name):
        self.Method_str = METHOD_TEMPLATE.format(name=name)

    def run(self):
        # Initialized orig_forward_inputs_list, orig_forward_returns_list, orig_forward_attrs_list
        self.CollectOriginalForwardInfo()

        if len(self.python_api_info) > 0:
            self.need_parse_python_api_args = True
            self.ParsePythonAPIInfo()
            for name in self.python_api_names:
                if "Tensor." in name:
                    api_name = name.split(".")[-1]
                    self.GenerateMethod(api_name)
                    self.api_name = api_name
                    break


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
        for forward_api_content in forward_api_list:
            f_generator = MethodGenerator(forward_api_content, None)
            status = f_generator.run()
            method_str = f_generator.Method_str
            if method_str != "":
                methods_map.append(
                    NAME_METHOD_MAPPING_TEMPLATE.format(
                        api_name=f_generator.api_name
                    )
                )
            self.MonkeyPatchTensorMethods_str += method_str
        result = ',\n '.join(methods_map)
        self.MonkeyPatchTensorMethods_str += METHODS_MAP_TEMPLATE.format(result)
        self.MonkeyPatchTensorMethods_str += FUNCTION_NAME_TEMPLATE.format(
            func_name="monkey_patch_generated_methods_for_tensor"
        )
        self.MonkeyPatchTensorMethods_str += SET_METHOD_TEMPLATE

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

    args = parser.parse_args()
    return args


def GenerateMonkeyPathFile(filepath, python_c_str):
    with open(filepath, 'w') as f:
        f.write(python_c_str)


if __name__ == "__main__":
    args = ParseArguments()
    api_yaml_path = args.api_yaml_path
    output_path = args.output_path
    gen = MonkeyPatchTensorMethodsGenerator(api_yaml_path)
    gen.run()
    GenerateMonkeyPathFile(output_path, gen.MonkeyPatchTensorMethods_str)
