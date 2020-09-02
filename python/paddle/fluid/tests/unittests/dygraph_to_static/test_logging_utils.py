#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import io
import logging
import os
import sys
import unittest

import gast
import six

import paddle
from paddle.fluid.dygraph.dygraph_to_static import logging_utils

# TODO(liym27): library mock needs to be installed separately in PY2,
#  but CI environment has not installed mock yet.
#  After discuss with Tian Shuo, now use mock only in PY3, and use it in PY2 after CI installs it.
if six.PY3:
    from unittest import mock
# else:
#     import mock


class TestLoggingUtils(unittest.TestCase):
    def setUp(self):
        self.verbosity_level = 1
        self.code_level = 3
        self.translator_logger = logging_utils._TRANSLATOR_LOGGER

    def test_verbosity(self):
        paddle.jit.set_verbosity(None)
        os.environ[logging_utils.VERBOSITY_ENV_NAME] = '3'
        self.assertEqual(logging_utils.get_verbosity(), 3)

        paddle.jit.set_verbosity(self.verbosity_level)
        self.assertEqual(self.verbosity_level, logging_utils.get_verbosity())

        # String is not supported
        with self.assertRaises(TypeError):
            paddle.jit.set_verbosity("3")

        with self.assertRaises(TypeError):
            paddle.jit.set_verbosity(3.3)

    def test_code_level(self):

        paddle.jit.set_code_level(None)
        os.environ[logging_utils.CODE_LEVEL_ENV_NAME] = '2'
        self.assertEqual(logging_utils.get_code_level(), 2)

        paddle.jit.set_code_level(self.code_level)
        self.assertEqual(logging_utils.get_code_level(), self.code_level)

        paddle.jit.set_code_level(9)
        self.assertEqual(logging_utils.get_code_level(), 9)

        with self.assertRaises(TypeError):
            paddle.jit.set_code_level(3.3)

    def test_log(self):
        stream = io.BytesIO() if six.PY2 else io.StringIO()
        log = self.translator_logger.logger
        stdout_handler = logging.StreamHandler(stream)
        log.addHandler(stdout_handler)

        warn_msg = "test_warn"
        error_msg = "test_error"
        log_msg_1 = "test_log_1"
        log_msg_2 = "test_log_2"

        if six.PY3:
            with mock.patch.object(sys, 'stdout', stream):
                logging_utils.warn(warn_msg)
                logging_utils.error(error_msg)
                self.translator_logger.verbosity_level = 1
                logging_utils.log(1, log_msg_1)
                logging_utils.log(2, log_msg_2)

            result_msg = '\n'.join([warn_msg, error_msg, log_msg_1, ""])
            self.assertEqual(result_msg, stream.getvalue())

    def test_log_transformed_code(self):
        source_code = "x = 3"
        ast_code = gast.parse(source_code)

        stream = io.BytesIO() if six.PY2 else io.StringIO()
        log = self.translator_logger.logger
        stdout_handler = logging.StreamHandler(stream)
        log.addHandler(stdout_handler)

        if six.PY3:
            with mock.patch.object(sys, 'stdout', stream):
                paddle.jit.set_code_level(1)
                logging_utils.log_transformed_code(1, ast_code,
                                                   "BasicApiTransformer")

                paddle.jit.set_code_level()
                logging_utils.log_transformed_code(
                    logging_utils.LOG_AllTransformer, ast_code,
                    "All Transformers")

            self.assertIn(source_code, stream.getvalue())


if __name__ == '__main__':
    unittest.main()
