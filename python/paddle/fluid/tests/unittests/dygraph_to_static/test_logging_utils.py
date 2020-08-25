#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import mock
import six
from paddle.fluid.dygraph.dygraph_to_static import logging_utils


class TestLoggingUtils(unittest.TestCase):
    def setUp(self):
        self.verbosity_level = 1
        self.code_level = 3
        self.translator_logger = logging_utils._TRANSLATOR_LOGGER

    def test_verbosity(self):
        logging_utils.set_verbosity(None)
        os.environ[logging_utils.VERBOSITY_ENV_NAME] = '3'
        self.assertEqual(logging_utils.get_verbosity(), 3)

        logging_utils.set_verbosity(self.verbosity_level)
        self.assertEqual(self.verbosity_level, logging_utils.get_verbosity())

        # String is not supported
        with self.assertRaises(ValueError):
            logging_utils.set_verbosity("3")

        with self.assertRaises(TypeError):
            logging_utils.set_verbosity(3.3)

    def test_code_level(self):

        logging_utils.set_code_level(self.code_level)
        self.assertEqual(logging_utils.get_code_level(), self.code_level)

        logging_utils.set_code_level(logging_utils.AssertTransformer)
        self.assertEqual(logging_utils.get_code_level(), 9)

        with self.assertRaises(TypeError):
            logging_utils.set_code_level(3.3)

    def test_log(self):
        stream = io.BytesIO() if six.PY2 else io.StringIO()
        log = self.translator_logger.logger
        stdout_handler = logging.StreamHandler(stream)
        log.addHandler(stdout_handler)

        warn_msg = "test_warn"
        error_msg = "test_error"
        log_msg_1 = "test_log_1"
        log_msg_2 = "test_log_2"
        with mock.patch.object(sys, 'stdout', stream):
            logging_utils.warn(warn_msg)
            logging_utils.error(error_msg)
            self.translator_logger.verbosity_level = 2
            logging_utils.log(1, log_msg_1)
            logging_utils.log(2, log_msg_2)

        result_msg = '\n'.join([warn_msg, error_msg, log_msg_2, ""])
        self.assertEqual(result_msg, stream.getvalue())

    def test_log_transformed_code(self):
        source_code = "x = 3"
        ast_code = gast.parse(source_code)

        stream = io.BytesIO() if six.PY2 else io.StringIO()
        log = self.translator_logger.logger
        stdout_handler = logging.StreamHandler(stream)
        log.addHandler(stdout_handler)

        with mock.patch.object(sys, 'stdout', stream):
            logging_utils.set_code_level(1)
            logging_utils.log_transformed_code(1, ast_code)

        self.assertIn(source_code, stream.getvalue())


if __name__ == '__main__':
    unittest.main()
