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
import shutil
import tempfile
from pathlib import Path

from pybind11_stubgen import (
    CLIArgs,
    Printer,
    Writer,
    run,
    stub_parser_from_args,
    to_output_and_subdir,
)


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
        if self.invalid_expr_as_ellipses:
            return "..."
        return f'"{invalid_expr.text}"'

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
        "--is-dir",
        default=False,
        action="store_true",
        help="If generate a dir instead of a file",
    )

    parser.add_argument(
        "--ignore-all-errors",
        default=False,
        action="store_true",
        help="Ignore all errors during module parsing",
    )

    args = parser.parse_args()

    return args


def generate_stub_file(
    output_dir,
    module_name,
    ignore_all_errors: bool = False,
    is_dir: bool = False,
):
    patch_pybind11_stubgen_printer()

    with tempfile.TemporaryDirectory() as tmpdirname:
        gen_stub(
            output_dir=tmpdirname,  # like: 'Paddle/python/',
            module_name=module_name,  # like: 'paddle.base.libpaddle',
            ignore_all_errors=ignore_all_errors,
        )
        paths = module_name.split('.')

        if is_dir:
            _path_dst = Path(output_dir).joinpath(paths[-1])
            if _path_dst.exists():
                shutil.rmtree(str(_path_dst))
        else:
            paths[-1] += '.pyi'
            _path_dst = Path(output_dir).joinpath(paths[-1])
            if _path_dst.exists():
                os.remove(str(_path_dst))

        shutil.move(str(Path(tmpdirname).joinpath(*paths)), output_dir)


def main():
    args = parse_args()
    generate_stub_file(
        output_dir=args.output_dir,
        module_name=args.module_name,
        ignore_all_errors=args.ignore_all_errors,
        is_dir=args.is_dir,
    )


if __name__ == '__main__':
    main()
