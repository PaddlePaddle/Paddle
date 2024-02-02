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

"""
Please check our Dy2St unittest dev guide for more details:
https://github.com/PaddlePaddle/Paddle/issues/61464

This script is used to check if the code is follow the Dy2St unittest dev guide.

Usage:

```bash
# check one file
python test/dygraph_to_static/check_approval.py test/dygraph_to_static/test_return.py

# check multiple files
python test/dygraph_to_static/check_approval.py test/dygraph_to_static/test_return.py test/dygraph_to_static/test_local_cast.py

# check whole directory
python test/dygraph_to_static/check_approval.py test/dygraph_to_static
```
"""
import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Location:
    lineno: int
    col_offset: int


class Diagnostic:
    def __init__(self, start: Location, end: Location, message: str):
        self.start = start
        self.end = end
        self.message = message


class UseToStaticAsDecoratorDiagnostic(Diagnostic):
    def __init__(self, start: Location, end: Location):
        super().__init__(
            start, end, 'Function should not use @paddle.jit.to_static directly'
        )


class TestClassInheritFromTestCaseDiagnostic(Diagnostic):
    def __init__(self, start: Location, end: Location):
        super().__init__(
            start,
            end,
            'Test class should inherit from Dy2StTestBase instead of unittest.TestCase',
        )


class TestCaseWithoutDecoratorDiagnostic(Diagnostic):
    def __init__(self, start: Location, end: Location):
        super().__init__(
            start,
            end,
            'Test case should use @test_legacy_and_pt_and_pir instead of no decorator',
        )


class TestCaseWithPirApiDecoratorDiagnostic(Diagnostic):
    def __init__(self, start: Location, end: Location):
        super().__init__(
            start,
            end,
            'Test case should use @test_legacy_and_pt_and_pir instead of @test_with_pir_api',
        )


ALLOW_LIST: dict[type[Diagnostic], list[str]] = {
    UseToStaticAsDecoratorDiagnostic: [
        "test_rollback.py",
        "test_legacy_error.py",
        "test_op_attr.py",
        "test_se_resnet.py",
        "test_lac.py",
        "test_convert_call.py",
        "test_local_cast.py",
        "test_origin_info.py",
        "test_full_name_usage.py",
        "test_pylayer.py",
    ],
    TestClassInheritFromTestCaseDiagnostic: [
        "test_function_spec.py",
        "test_setter_helper.py",
        "test_eval_frame.py",
        "test_ignore_module.py",
        "test_legacy_error.py",
        "test_local_cast.py",
        "test_ordered_set.py",
        "test_origin_info.py",
        "test_logging_utils.py",
        "test_move_cuda_pinned_tensor.py",
        "test_pylayer.py",
    ],
    TestCaseWithoutDecoratorDiagnostic: [
        "test_logical.py",
        "test_inplace_assign.py",
        # TODO: Remove these files from the allow list after it's support PIR mode
        "test_list.py",
        "test_bmn.py",
        "test_tensor_hook.py",
        "test_container.py",
        "test_to_tensor.py",
        "test_warning.py",
        "test_typing.py",
        "test_gradname_parse.py",
        "test_cache_program.py",
        "test_for_enumerate.py",
        "test_lac.py",
        "test_sentiment.py",
        "test_save_load.py",
        "test_cinn.py",
        "test_declarative.py",
        "test_fallback.py",
        "test_no_gradient.py",
    ],
    TestCaseWithPirApiDecoratorDiagnostic: [],
}


def is_test_class(node: ast.ClassDef):
    return node.name.startswith('Test')


def is_test_case(node: ast.FunctionDef):
    return node.name.startswith('test')


class Checker(ast.NodeVisitor):
    diagnostics: list[Diagnostic]

    def __init__(self):
        super().__init__()
        self.diagnostics = []


class TestBaseChecker(Checker):
    REGEX_TEST_WITH_PIR_API = re.compile(r".*test_with_pir_api")

    def visit_ClassDef(self, node: ast.ClassDef):
        if not is_test_class(node):
            return

        # Check if the test class inherits from unittest.TestCase
        for base in node.bases:
            if (
                isinstance(base, ast.Attribute)
                and isinstance(base.value, ast.Name)
                and base.value.id == 'unittest'
                and base.attr == 'TestCase'
            ) or (isinstance(base, ast.Name) and base.id == 'TestCase'):
                # print(f'Found test class {node.name}')
                start = Location(node.lineno, node.col_offset)
                end = Location(node.end_lineno, node.end_col_offset)  # type: ignore
                self.diagnostics.append(
                    TestClassInheritFromTestCaseDiagnostic(start, end)
                )
                return

            if (
                isinstance(base, ast.Attribute)
                and isinstance(base.value, ast.Name)
                and base.value.id == 'dygraph_to_static'
                and base.attr == 'Dy2StTestBase'
            ) or (isinstance(base, ast.Name) and base.id == 'Dy2StTestBase'):
                for sub_node in node.body:
                    if isinstance(sub_node, ast.FunctionDef) and is_test_case(
                        sub_node
                    ):
                        self.check_test_case(sub_node)
                return

        self.generic_visit(node)

    def check_test_case(self, node: ast.FunctionDef):
        # Check if the test case has not any decorator
        if not node.decorator_list:
            start = Location(node.lineno, node.col_offset)
            end = Location(node.end_lineno, node.end_col_offset)  # type: ignore
            self.diagnostics.append(
                TestCaseWithoutDecoratorDiagnostic(start, end)
            )
        # Check if the test case use @test_with_pir_api
        for decorator in node.decorator_list:
            decorator_str = ast.unparse(decorator).strip()
            if TestBaseChecker.REGEX_TEST_WITH_PIR_API.match(decorator_str):
                start = Location(node.lineno, node.col_offset)
                end = Location(node.end_lineno, node.end_col_offset)  # type: ignore
                self.diagnostics.append(
                    TestCaseWithPirApiDecoratorDiagnostic(start, end)
                )


class FunctionTostaticChecker(Checker):
    REGEX_TO_STATIC = re.compile(r"((paddle\.)?jit\.)?to_static(\(.+\))?")

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Function should not decorate with @paddle.jit.to_static directly
        for decorator in node.decorator_list:
            decoreator_name = ast.unparse(decorator).strip()
            if FunctionTostaticChecker.REGEX_TO_STATIC.match(decoreator_name):
                start = Location(node.lineno, node.col_offset)
                end = Location(node.end_lineno, node.end_col_offset)  # type: ignore
                self.diagnostics.append(
                    UseToStaticAsDecoratorDiagnostic(start, end)
                )


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Check if the code follow the Dy2St unittest dev guide.'
    )
    parser.add_argument(
        'files', type=str, nargs='+', help='files to be checked'
    )
    args = parser.parse_args()
    return args


def show_diagnostics(
    diagnostics: list[tuple[Path, list[Diagnostic]]],
    show_diagnostic_classes: tuple[type[Diagnostic], ...],
):
    total_errors = sum(
        len(file_diagnostics) for _, file_diagnostics in diagnostics
    )
    if not total_errors:
        return
    print(f'Total errors: {total_errors}')
    for file, file_diagnostics in diagnostics:
        if not file_diagnostics:
            continue
        for diagnostic in file_diagnostics:
            if not isinstance(diagnostic, show_diagnostic_classes):
                continue
            print(
                f'{file}:{diagnostic.start.lineno}:{diagnostic.start.col_offset}: {diagnostic.message}'
            )


def expand_glob(files) -> list[Path]:
    expanded = []
    for file in files:
        path = Path(file)
        if path.is_dir():
            expanded.extend(path.glob('**/test_*.py'))
        else:
            expanded.append(path)
    return expanded


def filter_diagnostics(diagnostics: list[tuple[Path, list[Diagnostic]]]):
    filtered_diagnostics = []
    for file, file_diagnostics in diagnostics:
        filtered_file_diagnostics = []
        for diagnostic in file_diagnostics:
            if type(diagnostic) not in ALLOW_LIST:
                filtered_file_diagnostics.append(diagnostic)
                continue
            if any(
                file.name == file_name
                for file_name in ALLOW_LIST[type(diagnostic)]
            ):
                continue
            filtered_file_diagnostics.append(diagnostic)
        if filtered_file_diagnostics:
            filtered_diagnostics.append((file, filtered_file_diagnostics))
    return filtered_diagnostics


def main():
    args = cli()
    files = args.files
    diagnostics: list[tuple[Path, list[Diagnostic]]] = []
    for file in expand_glob(files):
        with open(file, 'r') as f:
            code = f.read()
            tree = ast.parse(code)
            # print(tree)
            # print(ast.dump(tree, indent=2))

            checkers: list[Checker] = [
                TestBaseChecker(),
                FunctionTostaticChecker(),
            ]
            for checker in checkers:
                checker.visit(tree)
            diagnostics.append(
                (
                    file,
                    [
                        diagnostic
                        for checker in checkers
                        for diagnostic in checker.diagnostics
                    ],
                )
            )
    diagnostics = filter_diagnostics(diagnostics)
    show_diagnostics(
        diagnostics,
        (
            UseToStaticAsDecoratorDiagnostic,
            TestClassInheritFromTestCaseDiagnostic,
            TestCaseWithoutDecoratorDiagnostic,
            TestCaseWithPirApiDecoratorDiagnostic,
        ),
    )
    if diagnostics:
        sys.exit(1)


if __name__ == "__main__":
    main()
