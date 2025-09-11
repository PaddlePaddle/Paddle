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

# We type-check the `Example` codes from docstring, like:
# 1. checking from input `apis`
# > python type_checking.py paddle.abs paddle.abs_ paddle.sin
# 2. checking from spec, with increment api
# > python type_checking.py
# 3. checking from spec, with full apis
# > python type_checking.py --full-test
# `--full-test` and `apis` should not be set at the same time.

from __future__ import annotations

import argparse
import doctest
import os
import pathlib
import pty
import re
import subprocess
import sys
import tempfile
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from sampcd_processor_utils import (
    extract_code_blocks_from_docstr,
    get_docstring,
    init_logger as init_samplecode_logger,
)

COLOR_CYAN = '\033[96m'
COLOR_RED = '\033[91m'
COLOR_BOLD = '\033[1m'
COLOR_CLEAR = '\033[0m'


class TypeCheckingLogger:
    def __init__(self, debug: bool = False) -> None:
        self._debug = debug

    def set_debug(self, debug: bool) -> None:
        self._debug = debug

    def debug(self, msg: str) -> None:
        if self._debug:
            print(msg)

    def info(self, msg: str) -> None:
        print(msg)

    def error(self, msg: str) -> None:
        print(msg)


logger = TypeCheckingLogger()


class TypeChecker:
    style: str = 'google'

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def run_on_directory(
        self,
        dir: pathlib.Path,
        filename_to_codeblock_identifier: dict[str, str],
    ) -> tuple[dict[str, str], str] | None:
        pass

    @abstractmethod
    def print_summary(
        self,
        error_messages: dict[str, str],
        raw_summary: str,
        whl_error: list[str],
    ) -> None:
        pass


@dataclass
class TestResult:
    api_name: str
    msg: str
    fail: bool = False
    extra_info: dict[str, Any] = field(default_factory=dict)


def pty_run(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a command in a pseudo-terminal to capture colored output."""
    master_fd, slave_fd = pty.openpty()
    try:
        # Start subprocess with its stdout/stderr attached to the pty slave.
        # Do not use text=True here because we're passing raw fds; we'll decode
        # the bytes we read from master_fd ourselves.
        proc = subprocess.Popen(
            command,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
        )

        # Parent no longer needs the slave fd — close it so the child can
        # receive EOF properly when it exits.
        try:
            os.close(slave_fd)
            slave_fd = -1
        except OSError:
            pass

        stdout_chunks: list[str] = []
        while True:
            try:
                chunk = os.read(master_fd, 4096)
                if not chunk:
                    break
                stdout_chunks.append(chunk.decode(errors="ignore"))
            except OSError:
                break

        proc.wait()
        stdout = ''.join(stdout_chunks)
        return subprocess.CompletedProcess(
            args=command,
            returncode=proc.returncode,
            stdout=stdout,
            stderr=None,
        )
    finally:
        try:
            os.close(master_fd)
        except OSError:
            pass
        try:
            os.close(slave_fd)
        except OSError:
            pass


class MypyChecker(TypeChecker):
    REGEX_MYPY_ERROR_ITEM = re.compile(
        r'^(?P<filepath>.*\.py):(?P<lineno>\d+):((?P<colno>\d+):)? (?P<level>error|note):(?P<msg>.*)$'
    )
    REGEX_MYPY_ERROR_SUMMARY = re.compile(
        r'Found (?P<num_errors>\d+) errors? in (?P<num_files>\d+) files?'
    )
    REGEX_TRIM_COLOR = re.compile(r'\x1b\[[0-9;]*m')

    def __init__(
        self,
        config_file: str,
        cache_dir: str,
        debug: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.config_file = config_file
        self.cache_dir = cache_dir
        self.debug = debug
        super().__init__(*args, **kwargs)

    def _parse_output(
        self, output: str, filename_to_codeblock_identifier: dict[str, str]
    ) -> tuple[dict[str, str], str]:
        current_api = None
        results: dict[str, str] = {}
        summary = ''

        for line in output.splitlines():
            line_no_color = self.REGEX_TRIM_COLOR.sub('', line)
            if self.REGEX_MYPY_ERROR_SUMMARY.match(line_no_color.strip()):
                summary = line.strip()
                continue
            m = self.REGEX_MYPY_ERROR_ITEM.match(line_no_color)
            if m:
                filename = pathlib.Path(m.group('filepath')).stem
                if filename not in filename_to_codeblock_identifier:
                    raise ValueError(
                        f'Unknown filename {filename} in mypy output'
                    )
                current_api = filename_to_codeblock_identifier[filename]
                results[current_api] = (
                    results.get(current_api, '') + line + '\n'
                )
            else:
                assert current_api is not None, (
                    f'Cannot parse mypy output line: {line}'
                    ' (no preceding filename line)'
                )
                results[current_api] += line + '\n'
        assert summary, 'No summary found in mypy output'
        return results, summary

    def run_on_directory(
        self,
        dir: pathlib.Path,
        filename_to_codeblock_identifier: dict[str, str],
    ) -> tuple[dict[str, str], str] | None:
        res = pty_run(
            [
                sys.executable,
                '-m',
                'mypy',
                f'--config-file={self.config_file}',
                f'--cache-dir={self.cache_dir}',
                "--pretty",
                str(dir),
            ],
        )
        if res.returncode == 0:
            print(f'No type errors found in directory {dir}')
            return None
        logger.debug('>>> Mypy stdout:')
        logger.debug(res.stdout)
        logger.debug('>>> Mypy stderr:')
        logger.debug(res.stderr)
        return self._parse_output(res.stdout, filename_to_codeblock_identifier)

    def print_summary(
        self,
        error_messages: dict[str, str],
        raw_summary: str,
        whl_error: list[str],
    ) -> None:
        failed_apis = {
            codeblock_identifier.split(':')[0]
            for codeblock_identifier in error_messages.keys()
        }

        if whl_error is not None and whl_error:
            logger.info(f"{whl_error} is not in whl.")
            logger.info("")
            logger.info("Please check the whl package and API_PR.spec!")
            logger.info(
                "You can follow these steps in order to generate API.spec:"
            )
            logger.info("1. cd ${paddle_path}, compile paddle;")
            logger.info("2. pip install build/python/dist/(build whl package);")
            logger.info(
                "3. run 'python tools/print_signatures.py paddle > paddle/fluid/API.spec'."
            )

        if not failed_apis:
            logger.info(">>> Type checking is successful!")
            return

        for codeblock_identifier, msg in error_messages.items():
            logger.error(
                f"{COLOR_RED}{COLOR_BOLD}TYPE CHECKING FAILED{COLOR_CLEAR} in {COLOR_CYAN}{COLOR_BOLD}{codeblock_identifier}{COLOR_CLEAR}"
            )
            logger.error(msg)
        logger.error(">>> Mypy summary:")
        logger.error(raw_summary)
        logger.error(">>> Mistakes found in type checking!")
        logger.error(
            ">>> Please recheck the type annotations. Run `tools/type_checking.py` to check the typing issues:"
        )
        logger.error(
            "    $ python tools/type_checking.py "
            + " ".join(sorted(failed_apis))
        )
        logger.error(
            ">>> For more information: https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/style_guide_and_references/type_annotations_specification_cn.html"
        )


def parse_args() -> argparse.Namespace:
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description='run Sample Code Type Checking'
    )

    parser.add_argument('--debug', dest='debug', action="store_true")
    parser.add_argument(
        '--config-file',
        dest='config_file',
        type=str,
        default=None,
        help='config file for type checker',
    )
    parser.add_argument(
        '--cache-dir',
        dest='cache_dir',
        type=str,
        default=None,
        help='cache dir for mypy',
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument('apis', nargs='*', type=str, default=[])
    group.add_argument('--full-test', dest='full_test', action="store_true")

    args = parser.parse_args()
    return args


def codeblock_identifier_to_filename(codeblock_identifier: str) -> str:
    # convert codeblock_identifier to a valid filename
    return codeblock_identifier.replace('.', '_').replace(':', '__')


def preprocess_codeblock(codeblock: str) -> str:
    # skip checking when the codeblock startswith `>>> # type: ignore`
    codeblock_for_checking = []
    for line in codeblock.splitlines():
        if line.strip().startswith('>>> # type: ignore'):
            break
        codeblock_for_checking.append(line)
    codeblock_for_checking = '\n'.join(codeblock_for_checking)

    # remove `doctest` in the codeblock, or the module `doctest` cannot `get_examples`` correctly
    codeblock_for_checking = re.sub(
        r'#\s*x?doctest\s*:.*', '', codeblock_for_checking
    )

    # `get_examples` codes with `>>>` and `...` stripped
    _example_code = doctest.DocTestParser().get_examples(codeblock_for_checking)
    example_code = '\n'.join(
        [l for e in _example_code for l in e.source.splitlines()]
    )
    return example_code


def generate_code_snippets(
    type_checker: TypeChecker,
    dir: pathlib.Path,
    docstrings_to_test: dict[str, str],
) -> dict[str, str]:
    _test_style = (
        type_checker.style
        if type_checker.style in {'google', 'freeform'}
        else 'google'
    )
    google_style = _test_style == 'google'

    codeblocks: list[tuple[str, str]] = []
    filename_to_codeblock_identifier: dict[str, str] = {}
    for api_name, raw_docstring in docstrings_to_test.items():
        # we may extract more than one codeblocks from docstring.
        for codeblock in extract_code_blocks_from_docstr(
            raw_docstring, google_style=google_style
        ):
            codeblock_name = codeblock['name']
            codeblock_id = codeblock['id']
            codeblock_identifier = (
                f'{api_name}:{codeblock_name or codeblock_id}'
            )

            codeblocks.append(
                (
                    codeblock_identifier,
                    preprocess_codeblock(codeblock['codes']),
                )
            )

    for codeblock_identifier, codeblock in codeblocks:
        filename = codeblock_identifier_to_filename(codeblock_identifier)
        filename_to_codeblock_identifier[filename] = codeblock_identifier

        with (dir / f'{filename}.py').open('w', encoding='utf-8') as f:
            f.write(codeblock)

    return filename_to_codeblock_identifier


def get_test_results(
    type_checker: TypeChecker,
    docstrings_to_test: dict[str, str],
) -> tuple[dict[str, str], str] | None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = pathlib.Path(tmp_dir)

        logger.info(f">>> Store code snippets to {tmp_dir} ...")
        filename_to_codeblock_identifier = generate_code_snippets(
            type_checker, tmp_dir, docstrings_to_test
        )

        logger.info(">>> Preprocess code snippets and run type checker ...")
        results = type_checker.run_on_directory(
            tmp_dir, filename_to_codeblock_identifier
        )
    return results


def run_type_checker(
    args: argparse.Namespace, type_checker: TypeChecker
) -> None:
    # init logger for samplecode utils
    init_samplecode_logger(debug=args.debug)
    # init our logger
    logger.set_debug(args.debug)

    logger.info(
        "----------------Codeblock Type Checking Start--------------------"
    )

    logger.info(">>> Get docstring from api ...")
    filter_api = lambda api_name: 'libpaddle' in api_name
    docstrings_to_test, whl_error = get_docstring(
        full_test=args.full_test,
        filter_api=filter_api,
        apis=[(api, api) for api in args.apis],
    )
    results = get_test_results(type_checker, docstrings_to_test)

    if results is None:
        logger.info(">>> No type errors found.")
        return

    logger.info(">>> Print summary ...")
    error_messages, raw_summary = results
    type_checker.print_summary(
        error_messages=error_messages,
        raw_summary=raw_summary,
        whl_error=whl_error,
    )
    raise SystemExit(1)


if __name__ == '__main__':
    base_path = pathlib.Path(__file__).resolve().parent.parent

    args = parse_args()
    mypy_checker = MypyChecker(
        config_file=(
            args.config_file
            if args.config_file
            else str(base_path / 'pyproject.toml')
        ),
        cache_dir=str(
            args.cache_dir if args.cache_dir else (base_path / '.mypy_cache')
        ),
        debug=args.debug,
    )
    run_type_checker(args, mypy_checker)
