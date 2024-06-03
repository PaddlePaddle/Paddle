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

# We type-check the `Example` codes from docstring.

from __future__ import annotations

import argparse
import doctest
import pathlib
import re
from abc import abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from mypy import api as mypy_api
from sampcd_processor_utils import (
    extract_code_blocks_from_docstr,
    get_docstring,
    init_logger,
    log_exit,
    logger,
)


class TypeChecker:
    style: str = 'google'

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def run(self, api_name: str, codeblock: str) -> TestResult:
        pass

    @abstractmethod
    def print_summary(
        self, test_results: list[TestResult], whl_error: list[str]
    ) -> None:
        pass


@dataclass
class TestResult:
    api_name: str
    msg: str
    fail: bool = False
    extra_info: dict[str, Any] = field(default_factory=dict)


class MypyChecker(TypeChecker):
    def __init__(
        self, config_file: str, cache_dir: str, *args: Any, **kwargs: Any
    ) -> None:
        self.config_file = config_file
        self.cache_dir = cache_dir
        super().__init__(*args, **kwargs)

    def run(self, api_name: str, codeblock: str) -> TestResult:
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
        _example_code = doctest.DocTestParser().get_examples(
            codeblock_for_checking
        )
        example_code = '\n'.join(
            [l for e in _example_code for l in e.source.splitlines()]
        )

        normal_report, error_report, exit_status = mypy_api.run(
            [
                f'--config-file={self.config_file}',
                f'--cache-dir={self.cache_dir}',
                '-c',
                example_code,
            ]
        )

        logger.debug('-' * 20)
        logger.debug(f'>>> Type hints with api {api_name} start ...')
        logger.debug(example_code)
        logger.debug('>>> Results ...')
        logger.debug('>>> mypy normal_report is ...')
        logger.debug(normal_report)
        logger.debug('>>> mypy error_report is ...')
        logger.debug(error_report)
        logger.debug('>>> mypy exit_status is ...')
        logger.debug(exit_status)
        logger.debug(f'>>> Type hints with api {api_name} end...')

        return TestResult(
            api_name=api_name,
            msg='\n'.join([normal_report, error_report]),
            fail=exit_status != 0,
            extra_info={
                'normal_report': normal_report,
                'error_report': error_report,
                'exit_status': exit_status,
            },
        )

    def print_summary(
        self, test_results: list[TestResult], whl_error: list[str]
    ) -> None:
        is_fail = False

        logger.warning("----------------Check results--------------------")

        if whl_error is not None and whl_error:
            logger.warning("%s is not in whl.", whl_error)
            logger.warning("")
            logger.warning("Please check the whl package and API_PR.spec!")
            logger.warning(
                "You can follow these steps in order to generate API.spec:"
            )
            logger.warning("1. cd ${paddle_path}, compile paddle;")
            logger.warning(
                "2. pip install build/python/dist/(build whl package);"
            )
            logger.warning(
                "3. run 'python tools/print_signatures.py paddle > paddle/fluid/API.spec'."
            )
            for test_result in test_results:
                if test_result.fail:
                    logger.error(
                        ">>> In addition, mistakes found in type checking: %s",
                        test_result.api_name,
                    )
                    logger.error(test_result.msg)
            log_exit(1)

        else:
            for test_result in test_results:
                if test_result.fail:
                    is_fail = True

                    logger.error(test_result.api_name)
                    logger.error(test_result.msg)

                else:
                    logger.debug(test_result.api_name)
                    logger.debug(test_result.msg)

            if is_fail:
                logger.error(">>> Mistakes found in type checking!")
                logger.error(">>> Please recheck the type annotations.")
                log_exit(1)

        logger.warning(">>> Type checking is successful!")
        logger.warning("----------------End of the Check--------------------")


def parse_args() -> argparse.Namespace:
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description='run Sample Code Type Checking'
    )
    parser.add_argument('--debug', dest='debug', action="store_true")
    parser.add_argument(
        '--logf', dest='logf', type=str, default=None, help='file for logging'
    )
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
    parser.add_argument('--full-test', dest='full_test', action="store_true")

    args = parser.parse_args()
    return args


def get_test_results(
    type_checker: TypeChecker, docstrings_to_test: dict[str, str]
) -> list[TestResult]:
    _test_style = (
        type_checker.style
        if type_checker.style in {'google', 'freeform'}
        else 'google'
    )
    google_style = _test_style == 'google'

    api_names = []
    codeblocks = []
    for api_name, raw_docstring in docstrings_to_test.items():
        # we may extract more than one codeblocks from docsting.
        for codeblock in extract_code_blocks_from_docstr(
            raw_docstring, google_style=google_style
        ):
            codeblock_name = codeblock['name']
            codeblock_id = codeblock['id']

            api_names.append(f'{api_name}:{codeblock_name or codeblock_id}')
            codeblocks.append(codeblock['codes'])

    test_results = []
    with ProcessPoolExecutor() as exe:
        test_results = exe.map(
            type_checker.run, api_names, codeblocks, timeout=600
        )

    return list(test_results)


def run_type_checker(
    args: argparse.Namespace, type_checker: TypeChecker
) -> None:
    # init logger
    init_logger(debug=args.debug, log_file=args.logf)

    logger.info(
        "----------------Codeblock Type Checking Start--------------------"
    )

    logger.info(">>> Get docstring from api ...")
    filter_api = lambda api_name: 'libpaddle' in api_name
    docstrings_to_test, whl_error = get_docstring(
        full_test=args.full_test, filter_api=filter_api
    )

    logger.info(">>> Running type checker ...")
    test_results = get_test_results(type_checker, docstrings_to_test)

    logger.info(">>> Print summary ...")
    type_checker.print_summary(test_results, whl_error)


if __name__ == '__main__':
    base_path = pathlib.Path(__file__).resolve().parent.parent

    args = parse_args()
    mypy_checker = MypyChecker(
        config_file=(
            args.config_file
            if args.config_file
            else (base_path / 'pyproject.toml')
        ),
        cache_dir=(
            args.cache_dir if args.cache_dir else (base_path / '.mypy_cache')
        ),
    )
    run_type_checker(args, mypy_checker)
