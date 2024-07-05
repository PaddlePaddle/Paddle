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

import pathlib
import unittest

from mypy import api as mypy_api

FILE_PATH = pathlib.Path(__file__).resolve().parent
TEST_PATH = FILE_PATH / '_typing'
BASE_PATH = FILE_PATH.parent.parent
CONFIG_FILE = BASE_PATH / 'pyproject.toml'
CACHE_DIR = BASE_PATH / '.mypy_cache'


class _TestTyping:
    debug: bool = False
    config_file: str = CONFIG_FILE
    cache_dir: str = CACHE_DIR
    test_dir: str = ''

    def test_cases(self) -> None:
        normal_report, error_report, exit_status = mypy_api.run(
            (["--show-traceback"] if self.debug else [])
            + [
                f'--config-file={self.config_file}',
                f'--cache-dir={self.cache_dir}',
                str(self.test_dir),
            ]
        )
        if exit_status != 0:
            print('-' * 20)
            print(f'>>> test_dir: {self.test_dir}')
            print(f'>>> FILE_PATH: {FILE_PATH}')
            print(f'>>> TEST_PATH: {TEST_PATH}')
            print(f'>>> BASE_PATH: {BASE_PATH}')
            print(f'>>> CONFIG_FILE: {CONFIG_FILE}')
            print(f'>>> CACHE_DIR: {CACHE_DIR}')
            print('>>> normal_report ...')
            print(normal_report)
            print('>>> error_report ...')
            print(error_report)

        self.assertTrue(exit_status == 0)


class TestPass(unittest.TestCase, _TestTyping):
    def setUp(self) -> None:
        self.test_dir = TEST_PATH / 'pass'


class TestFail(unittest.TestCase, _TestTyping):
    def setUp(self) -> None:
        self.test_dir = TEST_PATH / 'fail'


class TestReveal(unittest.TestCase, _TestTyping):
    def setUp(self) -> None:
        self.test_dir = TEST_PATH / 'reveal'


if __name__ == '__main__':
    unittest.main()
