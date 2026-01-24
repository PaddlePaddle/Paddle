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

# Template for the entire generated file
FILE_TEMPLATE = '''# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""
Auto-generated tensor methods patch.

This module patches Tensor methods to use Python-layer APIs first,
ensuring compatibility with dy2static (dynamic-to-static) transformation.
"""

import paddle
from paddle import _C_ops
from paddle.tensor import magic_method_func
from .. import core


# List of method names to be patched on Tensor
_tensor_method_names = [
{tensor_methods}
]

# List of function names to be patched on paddle module
_paddle_func_names = [
{paddle_funcs}
]

# List of function names to be patched on paddle.nn.functional
_nn_func_names = [{nn_funcs}]


def _create_tensor_method(op_name):
    """Create a Tensor method wrapper for dy2static compatibility."""
    def method(self, *args, **kwargs):
        from paddle.base.framework import in_dynamic_or_pir_mode
        if in_dynamic_or_pir_mode():
            c_op = getattr(_C_ops, op_name, None)
            if c_op is not None:
                return c_op(self, *args, **kwargs)
        raise RuntimeError(
            f"Tensor.{{op_name}}() is not available in static graph mode. "
            f"Please implement paddle.tensor.{{op_name}}() with static graph support."
        )

    method.__name__ = op_name
    method.__qualname__ = f"Tensor.{{op_name}}"
    return method


def _create_paddle_func(op_name):
    """Create a paddle-level function wrapper for dy2static compatibility."""
    def func(*args, **kwargs):
        from paddle.base.framework import in_dynamic_or_pir_mode
        if in_dynamic_or_pir_mode():
            c_op = getattr(_C_ops, op_name, None)
            if c_op is not None:
                return c_op(*args, **kwargs)
        raise RuntimeError(
            f"paddle.{{op_name}}() is not available in static graph mode. "
            f"Please implement paddle.tensor.{{op_name}}() with static graph support."
        )

    func.__name__ = op_name
    func.__qualname__ = op_name
    return func


def _create_nn_func(op_name):
    """Create a nn.functional-level function wrapper for dy2static compatibility."""
    def func(*args, **kwargs):
        from paddle.base.framework import in_dynamic_or_pir_mode
        if in_dynamic_or_pir_mode():
            c_op = getattr(_C_ops, op_name, None)
            if c_op is not None:
                return c_op(*args, **kwargs)
        raise RuntimeError(
            f"paddle.nn.functional.{{op_name}}() is not available in static graph mode. "
            f"Please implement it with static graph support."
        )

    func.__name__ = op_name
    func.__qualname__ = f"nn.functional.{{op_name}}"
    return func


def monkey_patch_generated_methods_for_tensor():
    """Patch Tensor/paddle/nn.functional with Python wrappers for dy2static compatibility."""
    global _patched_methods, _patched_funcs, _patched_nn_funcs

    local_tensor = core.eager.Tensor
    magic_method_dict = {{v: k for k, v in magic_method_func}}

    # Patch Tensor methods
    for method_name in _tensor_method_names:
        method = _create_tensor_method(method_name)
        _patched_methods[method_name] = method
        setattr(local_tensor, method_name, method)
        magic_name = magic_method_dict.get(method_name)
        if magic_name:
            setattr(local_tensor, magic_name, method)

    # Patch paddle.tensor and paddle module functions
    for func_name in _paddle_func_names:
        func = _create_paddle_func(func_name)
        _patched_funcs[func_name] = func
        setattr(paddle.tensor, func_name, func)
        setattr(paddle, func_name, func)

    # Patch paddle.nn.functional functions
    for func_name in _nn_func_names:
        func = _create_nn_func(func_name)
        _patched_nn_funcs[func_name] = func
        setattr(paddle.nn.functional, func_name, func)

    # Update paddle.linalg aliases (imported before monkey_patch runs)
    _linalg_alias_map = {{'inverse': 'inv'}}
    for tensor_name, linalg_name in _linalg_alias_map.items():
        if tensor_name in _patched_funcs and hasattr(paddle, 'linalg'):
            setattr(paddle.linalg, linalg_name, _patched_funcs[tensor_name])


_patched_methods = {{}}
_patched_funcs = {{}}
_patched_nn_funcs = {{}}


def _get_methods_map():
    if _patched_methods:
        return list(_patched_methods.items())
    return [(name, _create_tensor_method(name)) for name in _tensor_method_names]


def _get_funcs_map():
    if _patched_funcs:
        return list(_patched_funcs.items())
    return [(name, _create_paddle_func(name)) for name in _paddle_func_names]


def _get_nn_funcs_map():
    if _patched_nn_funcs:
        return list(_patched_nn_funcs.items())
    return [(name, _create_nn_func(name)) for name in _nn_func_names]


class _LazyMapProxy:
    def __init__(self, getter):
        self._getter = getter
    def __iter__(self):
        return iter(self._getter())
    def __len__(self):
        return len(self._getter())


methods_map = _LazyMapProxy(_get_methods_map)
funcs_map = _LazyMapProxy(_get_funcs_map)
nn_funcs_map = _LazyMapProxy(_get_nn_funcs_map)
'''

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
        self.op_name = ""

    def run(self):
        # Initialized orig_forward_inputs_list, orig_forward_returns_list, orig_forward_attrs_list
        self.CollectOriginalForwardInfo()
        if len(self.python_api_info) > 0:
            self.need_parse_python_api_args = True
            self.ParsePythonAPIInfo()
            self.op_name = self.forward_api_name
            ClassifyAPIByPrefix(self.python_api_info, self.forward_api_name)


def ExtractPrefix(full_name):
    res = ""
    for m in full_name.split(".")[:-1]:
        res += m + '.'
    return res


def ClassifyAPIByPrefix(python_api_info, op_name):
    python_api_names = python_api_info["name"]
    for name in python_api_names:
        prefix = ExtractPrefix(name)
        if prefix == "paddle.":
            if op_name not in paddle_func_map:
                paddle_func_map.append(op_name)
        elif prefix == "paddle.Tensor.":
            if op_name not in tensor_method_map:
                tensor_method_map.append(op_name)
        elif prefix == "paddle.nn.functional.":
            if op_name not in nn_func_map:
                nn_func_map.append(op_name)
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
        forward_api_list = self.forward_api_list

        # Collect op names from ops.yaml
        for forward_api_content in forward_api_list:
            f_generator = MethodGenerator(forward_api_content, None)
            f_generator.run()

        # Collect op names from python_api_info.yaml
        for ops_name, python_api_info in python_api_info_from_yaml.items():
            ClassifyAPIByPrefix(python_api_info, ops_name)

        # Format method names list
        def format_names_list(names, indent=4):
            if not names:
                return ""
            lines = []
            line = " " * indent
            for i, name in enumerate(sorted(names)):
                item = f"'{name}'"
                if i < len(names) - 1:
                    item += ", "
                if len(line) + len(item) > 88:
                    lines.append(line.rstrip())
                    line = " " * indent + item
                else:
                    line += item
            if line.strip():
                lines.append(line.rstrip())
            return "\n".join(lines)

        # Generate final file content
        self.MonkeyPatchTensorMethods_str = FILE_TEMPLATE.format(
            tensor_methods=format_names_list(tensor_method_map),
            paddle_funcs=format_names_list(paddle_func_map),
            nn_funcs=", ".join(f"'{n}'" for n in sorted(nn_func_map))
            if nn_func_map
            else "",
        )

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
