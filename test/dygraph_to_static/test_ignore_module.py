#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import traceback
import unittest

import scipy

import paddle
from paddle.jit import ignore_module
from paddle.jit.dy2static.convert_call_func import BUILTIN_LIKELY_MODULES

logger = logging.getLogger(__file__)


def logging_warning():
    logging.warning('This is a warning message')
    logger.warning('This is a warning message')


class TestLoggingWarning(unittest.TestCase):
    def test_skip_ast_convert_logging_warning(self):
        static_logging_warning = paddle.jit.to_static(
            logging_warning, full_graph=True
        )
        static_logging_warning()


class TestIgnoreModule(unittest.TestCase):
    def test_ignore_module(self):
        modules = [scipy, traceback]
        ignore_module(modules)
        self.assertEqual(
            [scipy, traceback],
            BUILTIN_LIKELY_MODULES[-2:],
            'Failed to add modules that ignore transcription',
        )


if __name__ == '__main__':
    unittest.main()
