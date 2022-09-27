# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import os
import numpy as np
import shutil
import tempfile

from paddle.hapi.logger import setup_logger


class TestSetupLogger(unittest.TestCase):

    def setUp(self):
        self.save_dir = tempfile.mkdtemp()
        self.save_file = os.path.join(self.save_dir, 'logger.txt')

    def tearDown(self):
        shutil.rmtree(self.save_dir)

    def logger(self, output=None):
        setup_logger(output=output)

    def test_logger_no_output(self):
        self.logger()

    def test_logger_dir(self):
        self.logger(self.save_dir)

    def test_logger_file(self):
        self.logger(self.save_file)


if __name__ == '__main__':
    unittest.main()
