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
    GeneratorBase,
    ParsePythonAPIInfoFromYAML,
)

IMPORT_TEMPLATE = """
import importlib

import paddle
from paddle import _C_ops
from paddle.tensor import magic_method_func
from .. import core
"""

FUNCTION_NAME_TEMPLATE = """
def {func_name}():
"""

# Unified template entry format: ('module.path', 'method_name', _op_name)
UNIFIED_NAME_METHOD_MAPPING_TEMPLATE = (
    """  ('{module_path}', '{method_name}', _{op_name})"""
)

# Unified map template for all module paths
UNIFIED_FUNCS_MAP_TEMPLATE = """
# Unified map: (module_path, method_name, func) for all APIs
_all_funcs_map = [
{}
]

# Backward-compatible exports derived from unified map
methods_map = [(name, func) for path, name, func in _all_funcs_map if path == 'paddle.Tensor']
funcs_map = [(name, func) for path, name, func in _all_funcs_map if path == 'paddle']
nn_funcs_map = [(name, func) for path, name, func in _all_funcs_map if path == 'paddle.nn.functional']

"""

METHOD_TEMPLATE = """
def _{name}(*args, **kwargs):
    return _C_ops.{name}(*args, **kwargs)
"""

SET_UNIFIED_FUNCTION_TEMPLATE = """
    # set methods and functions for all modules using unified approach
    local_tensor = core.eager.Tensor
    magic_method_dict = {v: k for k, v in magic_method_func}

    for module_path, method_name, method in _all_funcs_map:
        try:
            # Special handling for paddle.Tensor (not a real module)
            if module_path == 'paddle.Tensor':
                setattr(local_tensor, method_name, method)
                magic_name = magic_method_dict.get(method_name)
                if magic_name:
                    setattr(local_tensor, magic_name, method)
                # Also set on paddle.tensor module
                setattr(paddle.tensor, method_name, method)
            else:
                module = importlib.import_module(module_path)
                setattr(module, method_name, method)
        except Exception as e:
            raise RuntimeError(
                f"Failed to set {method_name} on module {module_path}: {e}"
            )
"""

# Unified map: list of (module_path, method_name, func) for all module paths
unified_func_map = []
# The python api info in python_api_info.yaml
python_api_info_from_yaml = {}


def ExtractPrefix(full_name):
    res = ""
    for m in full_name.split(".")[:-1]:
        res += m + '.'
    return res


def GenerateMethod(name):
    return METHOD_TEMPLATE.format(name=name)


def ClassifyAPIByPrefix(python_api_info, op_name):
    """Classify API by prefix and add to unified map.

    All APIs are stored in a unified format: (module_path, method_name, func)
    """
    python_api_names = python_api_info["name"]
    for name in python_api_names:
        prefix = ExtractPrefix(name)
        method_name = name.split(".")[
            -1
        ]  # Extract the method name from full path

        if not prefix.startswith("paddle."):
            raise Exception("Unsupported Prefix " + prefix, "API : " + name)

        # Remove trailing dot to get module_path
        module_path = prefix.rstrip('.')
        unified_mapping = UNIFIED_NAME_METHOD_MAPPING_TEMPLATE.format(
            module_path=module_path,
            method_name=method_name,
            op_name=op_name,
        )
        unified_func_map.append(unified_mapping)


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

        method_str = ""
        # python api in python_api_info.yaml
        for ops_name, python_api_info in python_api_info_from_yaml.items():
            method_str += GenerateMethod(ops_name)
            ClassifyAPIByPrefix(python_api_info, ops_name)

        self.MonkeyPatchTensorMethods_str += method_str
        # Use unified map for all module paths
        result = ',\n '.join(unified_func_map)
        self.MonkeyPatchTensorMethods_str += UNIFIED_FUNCS_MAP_TEMPLATE.format(
            result
        )
        self.MonkeyPatchTensorMethods_str += FUNCTION_NAME_TEMPLATE.format(
            func_name="monkey_patch_generated_methods_for_tensor"
        )
        self.MonkeyPatchTensorMethods_str += SET_UNIFIED_FUNCTION_TEMPLATE

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
