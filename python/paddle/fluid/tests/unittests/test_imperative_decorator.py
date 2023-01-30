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
import inspect
import unittest

from test_imperative_base import new_program_scope

import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework


class TestTracerMode(unittest.TestCase):
=======
import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
import unittest
import inspect

from test_imperative_base import new_program_scope
from paddle.fluid.framework import _test_eager_guard


class TestTracerMode(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.init_mode = True

    def get_tracer_mode(self):
        assert fluid._non_static_mode(), "Dygraph mode must be enabled"

    @fluid.dygraph.no_grad
    def no_grad_func(self, a):
        self.assertEqual(self.tracer._has_grad, False)
        return a

    @framework.dygraph_not_support
    def not_support_func(self):
        return True

    def check_not_support_rlt(self, ans):
        try:
            rlt = self.not_support_func()
        except AssertionError:
            rlt = False
        finally:
            self.assertEqual(rlt, ans)

<<<<<<< HEAD
    def test_main(self):
=======
    def func_main(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        with fluid.dygraph.guard():
            self.tracer = framework._dygraph_tracer()
            self.tracer._train_mode = self.init_mode

            self.assertEqual(self.no_grad_func(1), 1)
            self.assertEqual(self.no_grad_func.__name__, "no_grad_func")

            def need_no_grad_func(a, b=1):
                return a + b

            decorated_func = fluid.dygraph.no_grad(need_no_grad_func)
            self.assertTrue(
<<<<<<< HEAD
                str(inspect.getfullargspec(decorated_func))
                == str(inspect.getfullargspec(need_no_grad_func))
            )
=======
                str(inspect.getfullargspec(decorated_func)) == str(
                    inspect.getfullargspec(need_no_grad_func)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.assertEqual(self.tracer._train_mode, self.init_mode)

        with fluid.dygraph.guard():
            self.check_not_support_rlt(False)

        paddle.enable_static()
        with new_program_scope():
            self.check_not_support_rlt(True)

<<<<<<< HEAD

class TestTracerMode2(TestTracerMode):
=======
    def test_main(self):
        with _test_eager_guard():
            self.func_main()
        self.func_main()


class TestTracerMode2(TestTracerMode):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.init_mode = False


class TestNoGradClass(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    @paddle.no_grad()
    def no_grad_func(self, a):
        self.assertEqual(self.tracer._train_mode, True)
        self.assertEqual(self.tracer._has_grad, False)
        return a

<<<<<<< HEAD
    def test_main(self):
=======
    def func_main(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        paddle.disable_static()

        self.tracer = framework._dygraph_tracer()
        self.tracer._train_mode = True

        self.assertEqual(self.no_grad_func(1), 1)
        self.assertEqual(self.no_grad_func.__name__, "no_grad_func")

        def need_no_grad_func(a, b=1):
            return a + b

        decorated_func = paddle.no_grad()(need_no_grad_func)
<<<<<<< HEAD
        self.assertEqual(
            str(inspect.getfullargspec(decorated_func)),
            str(inspect.getfullargspec(need_no_grad_func)),
        )
=======
        self.assertEqual(str(inspect.getfullargspec(decorated_func)),
                         str(inspect.getfullargspec(need_no_grad_func)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        def test_gen():
            for i in range(3):
                yield i

        a = 0
        for i in test_gen():
            a += i

        @paddle.no_grad()
        def test_wrapped_gen():
            for i in range(3):
                yield i

        b = 0
        for i in test_wrapped_gen():
            b += i

        self.assertEqual(a, b)

<<<<<<< HEAD
=======
    def test_main(self):
        with _test_eager_guard():
            self.func_main()
        self.func_main()

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

if __name__ == '__main__':
    unittest.main()
