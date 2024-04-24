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

import unittest

from mypy import api


class TestTypeHints(unittest.TestCase):
    def check_with_mypy(self, file_path):
        stdout, stderr, exitcode = api.run([file_path])
        assert exitcode == 0, stdout + '\n' + stderr

    def test_dtype(self):
        self.check_with_mypy("./dtype.py")

    def test_shape(self):
        self.check_with_mypy("./shape.py")

    def test_layout(self):
        self.check_with_mypy("./layout.py")

    def test_basic(self):
        self.check_with_mypy("./basic.py")

    def test_device(self):
        self.check_with_mypy("./device.py")


if __name__ == "__main__":
    unittest.main()
