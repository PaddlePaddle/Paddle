# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
<<<<<<< HEAD
import builtins
import unittest

from paddle.check_import_scipy import check_import_scipy
=======
import six.moves.builtins as builtins
from paddle.check_import_scipy import check_import_scipy
import unittest
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def my_import(name, globals=None, locals=None, fromlist=(), level=0):
    raise ImportError('DLL load failed, unittest: import scipy failed')


class importTest(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_import(self):
        testOsName = 'nt'
        old_import = builtins.__import__
        builtins.__import__ = my_import
        self.assertRaises(ImportError, check_import_scipy, testOsName)
        builtins.__import__ = old_import


if __name__ == '__main__':
    unittest.main()
