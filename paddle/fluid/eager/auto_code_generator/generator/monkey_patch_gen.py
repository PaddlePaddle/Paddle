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


GENERATED_OP_FUNC_TEMPLATE = """
def _{op_name}(*args, **kwargs):
    return _C_ops.{op_name}(*args, **kwargs)
"""

METHOD_OP_MAP_TEMPLATE = """  ('{module_path}', '{method_name}', _{op_name})"""

MONKYE_PATCH_TEMPLATE = """
# (module_path, method_name, _op_name) for all APIs
_all_method_op_map = [
{}
]

methods_map = [(name, func) for path, name, func in _all_method_op_map if path == 'paddle.Tensor']
funcs_map = [(name, func) for path, name, func in _all_method_op_map if path == 'paddle']
nn_funcs_map = [(name, func) for path, name, func in _all_method_op_map if path == 'paddle.nn.functional']

def monkey_patch_generated_methods_for_tensor():
    # set methods and functions for all modules using unified approach
    local_tensor = core.eager.Tensor
    magic_method_dict = {{v: k for k, v in magic_method_func}}

    for module_path, method_name, _op_name in _all_method_op_map:
        try:
            # Special handling for paddle.Tensor (not a real module)
            if module_path == 'paddle.Tensor':
                setattr(local_tensor, method_name, _op_name)
                magic_name = magic_method_dict.get(method_name)
                if magic_name:
                    setattr(local_tensor, magic_name, _op_name)
                # Also set on paddle.tensor module
                setattr(paddle.tensor, method_name, _op_name)
            else:
                module = importlib.import_module(module_path)
                setattr(module, method_name, _op_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to set {{method_name}} on module {{module_path}}: {{e}}"
            )
"""


def ExtractPrefix(full_name):
    res = ""
    for m in full_name.split(".")[:-1]:
        res += m + '.'
    return res


def ClassifyAPIByPrefix(python_api_info, op_name):
    """Classify API by prefix and add to unified map.

    All APIs are stored in a unified format: (module_path, method_name, func)
    """
    python_api_names = python_api_info["name"]
    method_op_map = []
    for name in python_api_names:
        prefix = ExtractPrefix(name)
        method_name = name.split(".")[
            -1
        ]  # Extract the method name from full path

        if not prefix.startswith("paddle."):
            raise Exception("Unsupported Prefix " + prefix, "API : " + name)

        # Remove trailing dot to get module_path
        module_path = prefix.rstrip('.')
        method_op_map.append(
            METHOD_OP_MAP_TEMPLATE.format(
                module_path=module_path,
                method_name=method_name,
                op_name=op_name,
            )
        )
    return method_op_map


class MonkeyPatchTensorMethodsGenerator(GeneratorBase):
    def __init__(self, api_yaml_path, python_api_info_yaml_path):
        # Parent members:
        # self.namespace
        # self.api_yaml_path
        # self.forward_api_list
        GeneratorBase.__init__(self, api_yaml_path)

        # Generated Result
        self.MonkeyPatchTensorMethods_str = ""
        self.python_api_info_yaml_path = python_api_info_yaml_path

    def GenerateMonkeyPatchTensorMethods(self):
        self.MonkeyPatchTensorMethods_str += IMPORT_TEMPLATE

        # list of (module_path, method_name, op)
        all_method_op_map = []
        # The python api info in python_api_info.yaml
        python_api_info_from_yaml = ParsePythonAPIInfoFromYAML(
            self.python_api_info_yaml_path
        )

        # python api in python_api_info.yaml
        for op_name, python_api_info in python_api_info_from_yaml.items():
            self.MonkeyPatchTensorMethods_str += (
                GENERATED_OP_FUNC_TEMPLATE.format(op_name=op_name)
            )
            all_method_op_map += ClassifyAPIByPrefix(python_api_info, op_name)

        self.MonkeyPatchTensorMethods_str += MONKYE_PATCH_TEMPLATE.format(
            ',\n '.join(all_method_op_map)
        )

    def run(self):
        # Read Yaml file
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

    gen = MonkeyPatchTensorMethodsGenerator(
        api_yaml_path, python_api_info_yaml_path
    )
    gen.run()
    GenerateMonkeyPathFile(output_path, gen.MonkeyPatchTensorMethods_str)
