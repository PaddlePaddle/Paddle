#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import paddle.fluid as fluid
=======
import paddle.fluid as fluid
import unittest
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


class TestContextManagerRaiseException(unittest.TestCase):
    # When exception raised in 'with' context, we should safely exit the context
    def test_func1(self):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def foo():
            with fluid.dygraph.guard():
                print("raise error in context manager")
                raise TypeError("error")

        self.assertRaises(TypeError, foo)

    def test_func2(self):
        # After test_func1 executed, if fluid.dygraph.guard() in test_func1 safely exited,
        # fluid._non_static_mode() should be false.
        self.assertEqual(fluid._non_static_mode(), False)


if __name__ == '__main__':
    unittest.main()
