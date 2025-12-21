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

"""
Tool to check if the code-block directives in the given files follow proper format.

The following format issues are checked:

- The code under code-block directive should be indented properly.
- The code-block directive should be followed by a blank line.

You can run this script as follows:

    python tools/check_code_block_format.py <file1> <file2> ...

If you want to check all Python and C++ files under a directory, you can provide the directory path as an argument:

    python tools/check_code_block_format.py <directory_path>

For example:

    python tools/check_code_block_format.py python paddle
"""

import argparse
import io
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

REGEX_CODE_BLOCK = re.compile(r"(?P<indent>\s*)..\s+code-block::\s*(\w+)?")
REGEX_CODE_BLOCK_ATTRS = re.compile(r"(?P<indent>\s*):(\w+):\s*(.*)?")


def with_color(text: str, color_code: str) -> str:
    return f"\033[{color_code}m{text}\033[0m"


def with_red(text: str) -> str:
    return with_color(text, "31")


def with_cyan(text: str) -> str:
    return with_color(text, "36")


def with_blue(text: str) -> str:
    return with_color(text, "34")


@dataclass
class CodeBlock:
    lines: list[str]
    start_lineno: int


@dataclass
class CodeBlockModifySuggestion:
    lineno: int
    message: str


class Diagnostic:
    def __init__(self, start: int, message: str):
        self.start = start
        self.message = message


def is_blank_line(line: str) -> bool:
    return line.strip() == ""


class CodeBlockIndentationDiagnostic(Diagnostic):
    def __init__(
        self,
        start: int,
        code_block: CodeBlock,
        suggestions: list[CodeBlockModifySuggestion] = [],
    ):
        super().__init__(
            start,
            "The code under code-block directive should be indented properly. The issue found in:\n"
            + format_code_block_with_suggestions(code_block, suggestions),
        )


class CodeBlockFollowsBlankLineDiagnostic(Diagnostic):
    def __init__(
        self,
        start: int,
        code_block: CodeBlock,
        suggestions: list[CodeBlockModifySuggestion] = [],
    ):
        super().__init__(
            start,
            "The code-block directive should be followed by a blank line. The issue found in:\n"
            + format_code_block_with_suggestions(code_block, suggestions),
        )


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Check if the code-block follows a blank line format.'
    )
    parser.add_argument(
        'files', type=str, nargs='+', help='files to be checked'
    )
    args = parser.parse_args()
    return args


def expand_glob(files) -> list[Path]:
    expanded = []
    for file in files:
        path = Path(file)
        if path.is_dir():
            expanded.extend(path.glob('**/*.py'))
            expanded.extend(path.glob('**/*.cc'))
            expanded.extend(path.glob('**/*.h'))
        else:
            expanded.append(path)
    return expanded


def format_code_block_with_suggestions(
    code_block: CodeBlock, suggestions: list[CodeBlockModifySuggestion]
) -> str:
    formatted_lines = []
    for line_offset, line in enumerate(code_block.lines):
        lineno_str = with_cyan(f"{code_block.start_lineno + line_offset:>5}|")
        formatted_lines.append(f"{lineno_str} {line}")
        for suggestion in suggestions:
            if suggestion.lineno == line_offset:
                formatted_lines.append(
                    f"     | {with_red('^')} {with_blue(suggestion.message)}\n"
                )
    return ''.join(formatted_lines)


def extract_code_blocks(
    file: io.TextIOWrapper,
) -> Iterator[CodeBlock]:
    lines_iter = iter(enumerate(file))

    while True:
        next_item = next(lines_iter, None)
        if next_item is None:
            break
        lineno, line = next_item
        match = REGEX_CODE_BLOCK.match(line)
        if match:
            indent = match.group('indent')
            indent_size = len(indent)
            code_block_start_lineno = lineno + 1
            code_block_lines = [line]
            indent_regex = re.compile(rf"\s{{{indent_size + 1}}}")
            while True:
                next_item = next(lines_iter, None)
                if next_item is None:
                    break
                lineno, line = next_item
                if not indent_regex.match(line) and not is_blank_line(line):
                    break
                code_block_lines.append(line)
            yield CodeBlock(code_block_lines, code_block_start_lineno)


def check_code_block_indentation(
    code_block: CodeBlock,
) -> list[Diagnostic]:
    suggestion = CodeBlockModifySuggestion(
        lineno=0, message="Next line should be indented."
    )
    diagnostic = CodeBlockIndentationDiagnostic(
        code_block.start_lineno,
        code_block,
        [suggestion],
    )
    if len(code_block.lines) == 1:
        suggestion.lineno = 1
        return [diagnostic]
    for line_offset, line in enumerate(code_block.lines[1:], 1):
        if not is_blank_line(line) and not REGEX_CODE_BLOCK_ATTRS.match(line):
            return []
        suggestion.lineno = line_offset
    return [diagnostic]


def check_code_block_follows_blank_line(
    code_block: CodeBlock,
) -> list[Diagnostic]:
    suggestion = CodeBlockModifySuggestion(
        lineno=0, message="Add a blank line here."
    )
    diagnostic = CodeBlockFollowsBlankLineDiagnostic(
        code_block.start_lineno,
        code_block,
        [suggestion],
    )
    if len(code_block.lines) < 2:
        return []

    def check_lines_starts_with_blank_line(
        lines: list[str], start_lineno: int
    ) -> bool:
        if not lines:
            return True
        first_line = lines[0]
        if REGEX_CODE_BLOCK_ATTRS.match(first_line):
            return check_lines_starts_with_blank_line(
                lines[1:], start_lineno + 1
            )
        suggestion.lineno = start_lineno
        return is_blank_line(first_line)

    if not check_lines_starts_with_blank_line(code_block.lines[1:], 1):
        return [diagnostic]
    return []


def check_code_block_format(file: io.TextIOWrapper) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for code_block in extract_code_blocks(file):
        diagnostics.extend(check_code_block_indentation(code_block))
        diagnostics.extend(check_code_block_follows_blank_line(code_block))
    return diagnostics


def show_diagnostic(file: Path, diagnostic: Diagnostic) -> None:
    print(
        f"{with_blue(str(file))}:{with_cyan(str(diagnostic.start))}: "
        f"{with_red('error:')} {diagnostic.message}"
    )


def main():
    args = cli()
    files = args.files
    diagnostics: list[tuple[Path, list[Diagnostic]]] = []
    for file in expand_glob(files):
        with open(file, 'r') as f:
            file_diagnostics = check_code_block_format(f)
            for diagnostic in file_diagnostics:
                diagnostics.append((file, file_diagnostics))
    if diagnostics:
        for file, file_diagnostics in diagnostics:
            for diagnostic in file_diagnostics:
                show_diagnostic(file, diagnostic)
        sys.exit(1)


if __name__ == "__main__":
    main()
