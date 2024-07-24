# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import argparse
import functools
import inspect
import keyword
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import yaml
from pybind11_stubgen import (
    CLIArgs,
    Printer,
    Writer,
    run,
    stub_parser_from_args,
    to_output_and_subdir,
)

PYBIND11_MAPPING = {
    'capsule': 'typing_extensions.CapsuleType',
    'paddle::Tensor': 'Tensor',
}

INPUT_TYPES_MAP = {
    'Tensor': 'Tensor',
    'Tensor[]': 'list[Tensor]',
}
ATTR_TYPES_MAP = {
    'IntArray': 'list[int]',
    'Scalar': 'float',
    'Scalar(int)': 'int',
    'Scalar(int64_t)': 'int',
    'Scalar(float)': 'float',
    'Scalar(double)': 'float',
    'Scalar[]': 'list[float]',
    'int': 'int',
    'int32_t': 'int',
    'int64_t': 'int',
    'long': 'float',
    'size_t': 'int',
    'float': 'float',
    'float[]': 'list[float]',
    'double': 'float',
    'bool': 'bool',
    'bool[]': 'list[bool]',
    'str': 'str',
    'str[]': 'list[str]',
    'Place': 'paddle._typing.PlaceLike',
    'DataLayout': 'paddle._typing.DataLayoutND',
    'DataType': 'paddle._typing.DTypeLike',
    'int64_t[]': 'list[int]',
    'int[]': 'list[int]',
}
OPTIONAL_TYPES_TRANS = {
    'Tensor': 'Tensor',
    'Tensor[]': 'list[Tensor]',
    'int': 'int',
    'int32_t': 'int',
    'int64_t': 'int',
    'float': 'float',
    'double': 'float',
    'bool': 'bool',
    'Place': 'paddle._typing.PlaceLike',
    'DataLayout': 'paddle._typing.DataLayoutND',
    'DataType': 'paddle._typing.DTypeLike',
}

PATTERN_FUNCTION = re.compile(r'^def (?P<name>.*)\(\*args, \*\*kwargs\):')
FUNCTION_VALUE_TRANS = {
    'true': 'True',
    'false': 'False',
}
OPS_YAML_IMPORTS = ['import paddle\n', 'from paddle import Tensor\n']


def patch_pybind11_stubgen_printer():
    # patch name with suffix '_' if `name` is a keyword like `in` to `in_`
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for arg in args:
                if hasattr(arg, 'name') and keyword.iskeyword(arg.name):
                    arg.name += '_'
            for k, w in kwargs.items():
                if hasattr(w, 'name') and keyword.iskeyword(arg.name):
                    kwargs[k].name += '_'
            return func(*args, **kwargs)

        return wrapper

    for name, value in inspect.getmembers(Printer):
        if inspect.isfunction(value) and name.startswith('print_'):
            setattr(Printer, name, decorator(getattr(Printer, name)))

    # patch invalid exp with `"xxx"` as a `typing.Any`
    def print_invalid_exp(self, invalid_expr) -> str:
        _text = invalid_expr.text
        return PYBIND11_MAPPING.get(_text, f'"{_text}"')

    Printer.print_invalid_exp = print_invalid_exp


def gen_stub(
    output_dir: str, module_name: str, ignore_all_errors: bool = False
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s - [%(levelname)7s] %(message)s",
    )

    args = CLIArgs(
        output_dir=output_dir,
        root_suffix=None,
        ignore_invalid_expressions=None,
        ignore_invalid_identifiers=None,
        ignore_unresolved_names=None,
        ignore_all_errors=ignore_all_errors,
        enum_class_locations=[],
        numpy_array_wrap_with_annotated=False,
        numpy_array_use_type_var=False,
        numpy_array_remove_parameters=False,
        print_invalid_expressions_as_is=True,
        print_safe_value_reprs=None,
        exit_code=False,
        dry_run=False,
        stub_extension='pyi',
        module_name=module_name,
    )

    parser = stub_parser_from_args(args)
    printer = Printer(
        invalid_expr_as_ellipses=not args.print_invalid_expressions_as_is
    )

    out_dir, sub_dir = to_output_and_subdir(
        output_dir=args.output_dir,
        module_name=args.module_name,
        root_suffix=args.root_suffix,
    )

    run(
        parser,
        printer,
        args.module_name,
        out_dir,
        sub_dir=sub_dir,
        dry_run=args.dry_run,
        writer=Writer(stub_ext=args.stub_extension),
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="python/paddle/",
    )
    parser.add_argument(
        "-m",
        "--module-name",
        type=str,
        default="",
    )

    parser.add_argument(
        "--ignore-all-errors",
        default=False,
        action="store_true",
        help="Ignore all errors during module parsing",
    )

    parser.add_argument(
        "--ops-yaml",
        nargs='*',
        help="Parse ops yaml, the input should be `<yaml path>;<ops module>` or `<yaml path>;<ops module>;<op_prefix>`"
        "like `/foo/bar/ops.yaml;paddle.x.y.ops` or /foo/bar/ops.yaml;paddle.x.y.ops;sparse",
    )

    args = parser.parse_args()

    return args


def _parse_yaml_inputs(
    ops_yaml: list[str],
) -> Generator[tuple[str, str, str], None, None]:
    for ops in ops_yaml:
        _ops = ops.split(';')
        if len(_ops) == 2:
            yaml_path, dst_module = _ops
            op_prefix = None
        elif len(_ops) == 3:
            yaml_path, dst_module, op_prefix = _ops
        yield yaml_path, dst_module, op_prefix


def _parse_dst_module_to_path(
    base_dir: str, module_name: str, dst_module: str
) -> str:
    assert dst_module.startswith(
        module_name
    )  # e.g.: `paddle.base.libpaddle` in `paddle.base.libpaddle.eager.ops`

    paths = dst_module.split('.')
    dst_path = Path(base_dir).joinpath(*paths)
    if dst_path.is_dir():
        dst_path = dst_path.joinpath('__init__.pyi')
    else:
        dst_path = dst_path.parent.joinpath(paths[-1] + '.pyi')

    assert dst_path.exists()
    return str(dst_path)


def generate_stub_file(
    output_dir,
    module_name,
    ignore_all_errors: bool = False,
    ops_yaml: list[str] | None = None,
):
    # patch `pybind11-stubgen`
    patch_pybind11_stubgen_printer()

    # generate stub files
    with tempfile.TemporaryDirectory() as tmpdirname:
        # gen stub
        gen_stub(
            output_dir=tmpdirname,  # e.g.: 'Paddle/python/',
            module_name=module_name,  # e.g.: 'paddle.base.libpaddle',
            ignore_all_errors=ignore_all_errors,
        )

        # parse ops yaml into file
        if ops_yaml is not None:
            _import_inserted = set()
            for yaml_path, dst_module, op_prefix in _parse_yaml_inputs(
                ops_yaml
            ):
                dst_module_path = _parse_dst_module_to_path(
                    tmpdirname,
                    module_name,
                    dst_module,
                )
                parse_yaml_ops(yaml_path, dst_module_path, op_prefix)

                if dst_module_path not in _import_inserted:
                    # insert imports once
                    insert_yaml_imports(dst_module_path)
                    _import_inserted.add(dst_module_path)

        # move stub files into output_dir
        paths = module_name.split('.')
        source_path = Path(tmpdirname).joinpath(*paths)

        if source_path.is_dir():
            _path_dst = Path(output_dir).joinpath(paths[-1])
            if _path_dst.exists():
                shutil.rmtree(str(_path_dst))
        else:
            paths[-1] += '.pyi'
            _path_dst = Path(output_dir).joinpath(paths[-1])
            if _path_dst.exists():
                os.remove(str(_path_dst))

        shutil.move(str(source_path), output_dir)


# ref: paddle/phi/api/generator/api_base.py
def _parse_input_and_attr(api_name, args_config, optional_vars=[]):
    inputs = {'names': [], 'input_info': {}}
    attrs = {'names': [], 'attr_info': {}}
    args_str = args_config.strip()
    assert args_str.startswith('(') and args_str.endswith(
        ')'
    ), f"Args declaration should start with '(' and end with ')', please check the args of {api_name} in yaml."
    args_str = args_str[1:-1]
    patten = re.compile(r',(?![^{]*\})')  # support int[] a={1,3}
    args_list = re.split(patten, args_str.strip())
    args_list = [x.strip() for x in args_list]

    for item in args_list:
        item = item.strip()
        type_and_name = item.split(' ')
        # match the input tensor
        has_input = False
        for in_type_symbol, in_type in INPUT_TYPES_MAP.items():
            if type_and_name[0] == in_type_symbol:
                input_name = type_and_name[1].strip()
                assert (
                    len(input_name) > 0
                ), f"The input tensor name should not be empty. Please check the args of {api_name} in yaml."
                assert (
                    len(attrs['names']) == 0
                ), f"The input Tensor should appear before attributes. please check the position of {api_name}:input({input_name}) in yaml"

                if input_name in optional_vars:
                    in_type = OPTIONAL_TYPES_TRANS[in_type_symbol]

                inputs['names'].append(input_name)
                inputs['input_info'][input_name] = in_type
                has_input = True
                break
        if has_input:
            continue

        # match the attribute
        for attr_type_symbol, attr_type in ATTR_TYPES_MAP.items():
            if type_and_name[0] == attr_type_symbol:
                attr_name = item[len(attr_type_symbol) :].strip()
                assert (
                    len(attr_name) > 0
                ), f"The attribute name should not be empty. Please check the args of {api_name} in yaml."
                default_value = None
                if '=' in attr_name:
                    attr_infos = attr_name.split('=')
                    attr_name = attr_infos[0].strip()
                    default_value = attr_infos[1].strip()

                if attr_name in optional_vars:
                    attr_type = OPTIONAL_TYPES_TRANS[attr_type_symbol]

                attrs['names'].append(attr_name)
                attrs['attr_info'][attr_name] = (attr_type, default_value)
                break

    return inputs, attrs


# ref: paddle/phi/api/generator/api_base.py
def _parse_output(api_name, output_config):
    def parse_output_item(output_item):
        output_type_map = {
            'Tensor': 'Tensor',
            'Tensor[]': 'list[Tensor]',
        }
        result = re.search(
            r"(?P<out_type>[a-zA-Z0-9_[\]]+)\s*(?P<name>\([a-zA-Z0-9_@]+\))?\s*(?P<expr>\{[^\}]+\})?",
            output_item,
        )
        assert (
            result is not None
        ), f"{api_name} : the output config parse error."
        out_type = result.group('out_type')
        assert (
            out_type in output_type_map
        ), f"{api_name} : Output type error: the output type only support Tensor and Tensor[], \
                but now is {out_type}."

        out_name = (
            'out'
            if result.group('name') is None
            else result.group('name')[1:-1]
        )
        out_size_expr = (
            None if result.group('expr') is None else result.group('expr')[1:-1]
        )
        return output_type_map[out_type], out_name, out_size_expr

    temp_list = output_config.split(',')

    if len(temp_list) == 1:
        out_type, out_name, size_expr = parse_output_item(temp_list[0])
        return [out_type], [out_name], [size_expr]
    else:
        out_type_list = []
        out_name_list = []
        out_size_expr_list = []
        for output_item in temp_list:
            out_type, out_name, size_expr = parse_output_item(output_item)
            out_type_list.append(out_type)
            out_name_list.append(out_name)
            out_size_expr_list.append(size_expr)

        return out_type_list, out_name_list, out_size_expr_list


def _make_sig_name(name):
    # 'lambda' -> 'lambda_'
    if keyword.iskeyword(name):
        name += '_'

    return name


def _make_attr(info):
    info_name, info_value = info
    if info_value is None:
        return info_name

    if info_name.startswith('list') and '{' in info_value:
        info_value = info_value.replace('{', '[').replace('}', ']')

    elif info_value in FUNCTION_VALUE_TRANS:
        info_value = FUNCTION_VALUE_TRANS[info_value]

    elif info_name == 'float' and info_value.lower().endswith('f'):
        info_value = info_value[:-1]

    elif info_name == 'int' and info_value.lower().endswith('l'):
        info_value = info_value[:-1]

    else:
        try:
            eval(info_value)
        except:
            info_value = f'"{info_value}"'

    return info_name + ' = ' + info_value


def _make_sig(name, sig):
    return _make_sig_name(name) + ': ' + _make_attr(sig)


def _make_op_function(name, inputs, attrs, output_type_list) -> str:
    input_info_names = inputs['names']
    input_info = inputs['input_info']
    attr_info_names = attrs['names']
    attr_info = attrs['attr_info']

    _sig_info = [
        _make_sig(_name, (input_info[_name], None))
        for _name in input_info_names
        if _name in input_info
    ]
    _sig_attr = [
        _make_sig(_name, attr_info[_name])
        for _name in attr_info_names
        if _name in attr_info
    ]
    sig_input = ', '.join(_sig_info + _sig_attr)
    sig_output = (
        output_type_list[0]
        if len(output_type_list) == 1
        else f'tuple[{", ".join(output_type_list)}]'
    )

    return f'def {name}({sig_input}) -> {sig_output}:\n'


# ref: paddle/phi/api/generator/api_base.py
def parse_yaml_ops(
    yaml_file: str, dst_module_path: str, op_prefix: str | None = None
) -> None:
    ops_names = {}
    ops_file = []
    import_line_no = 0
    with open(dst_module_path) as f:
        for line_no, line in enumerate(f.readlines()):
            match_obj = PATTERN_FUNCTION.match(line)
            if match_obj is not None:
                ops_names[match_obj.group('name')] = line_no
            ops_file.append(line)

    with open(yaml_file) as f:
        api_list = yaml.load(f, Loader=yaml.FullLoader)
        for api_item_yaml in api_list:
            optional_vars = []
            if 'optional' in api_item_yaml:
                optional_vars = [
                    item.strip()
                    for item in api_item_yaml['optional'].split(',')
                ]

            op_name = api_item_yaml['op']
            op_name = (
                f'{op_prefix}_{op_name}' if op_prefix is not None else op_name
            )
            op_args = api_item_yaml['args']
            op_output = api_item_yaml['output']
            op_inputs, op_attrs = _parse_input_and_attr(
                op_name, op_args, optional_vars
            )
            output_type_list, _, _ = _parse_output(op_name, op_output)

            # op and inplace op
            for _op_name in [op_name, op_name + '_']:
                if _op_name in ops_names:
                    try:
                        ops_file[ops_names[_op_name]] = _make_op_function(
                            _op_name, op_inputs, op_attrs, output_type_list
                        )
                    except:
                        print(_op_name, op_inputs, op_attrs, output_type_list)
                        raise

    with open(dst_module_path, 'w') as f:
        f.writelines(ops_file)


def insert_yaml_imports(dst_module_path):
    ops_file = []
    with open(dst_module_path, 'r') as f:
        ops_file = f.readlines()

    import_line_no = 0
    for line_no, line in enumerate(ops_file):
        if line.startswith('from __future__ import annotations'):
            import_line_no = line_no + 1
            break

    # insert imports
    ops_file = (
        ops_file[:import_line_no] + OPS_YAML_IMPORTS + ops_file[import_line_no:]
    )

    with open(dst_module_path, 'w') as f:
        f.writelines(ops_file)


def main():
    args = parse_args()
    generate_stub_file(
        output_dir=args.output_dir,
        module_name=args.module_name,
        ignore_all_errors=args.ignore_all_errors,
        ops_yaml=args.ops_yaml,
    )


if __name__ == '__main__':
    main()
