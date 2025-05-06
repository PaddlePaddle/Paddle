# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import unittest

from test_case_base import (
    TestCaseBase,
)

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph
from paddle.jit.sot.utils import strict_mode_guard

NOT_ALLOW_FALLBACK = sys.version_info < (3, 11) and sys.version_info >= (3, 9)


class TestNestingCase(TestCaseBase):
    @strict_mode_guard(NOT_ALLOW_FALLBACK)
    @check_no_breakgraph
    def test_try_nesting(self):
        def try_nesting_wo_error(x):
            try:
                try:
                    try:
                        try:
                            try:
                                try:
                                    x -= 1
                                    raise ValueError(
                                        "TESTING"
                                    )  # RAISE_VARARGS(1)
                                    x += 2
                                except NotImplementedError:
                                    x /= 3
                                    raise  # RAISE_VARARGS(0)
                            except (KeyError, IndexError):
                                x *= 4
                            except ValueError:
                                x += 5
                                raise NameError  # RAISE_VARARGS(1)
                        except SyntaxError:
                            x /= 6
                        except (TypeError, FileNotFoundError, NameError) as e:
                            x -= 7
                            raise TimeoutError(
                                "TESTING"
                            ) from e  # RAISE_VARARGS(2)
                    except:
                        x /= 8
                        raise AssertionError
                except IndentationError as e:
                    x *= 9
                except AssertionError as e:
                    x += 10
                    raise  # RAISE_VARARGS(0)
            except Exception as e:
                x /= 11

            return x + 12

        self.assert_results(try_nesting_wo_error, paddle.to_tensor(0.5))

    @strict_mode_guard(NOT_ALLOW_FALLBACK)
    @check_no_breakgraph
    def test_function_nesting(self):
        def raise_value_error_obj(x):
            x += 1
            raise ValueError("")

        def raise_value_error_cls(x):
            x += 2
            raise ValueError

        def raise_zero_div_error(x):
            x += 3
            return 19.0 / 0

        def raise_assert_error(x):
            x += 4
            assert []

        def raise_not_implemented_error(x):
            x += 5
            raise NotImplementedError

        def one_nesting(x, func):
            x *= 6
            func(x)

        def two_nesting(x, func):
            x /= 7
            one_nesting(x, func)

        def three_nesting(x, func):
            x -= 8
            two_nesting(x, func)

        def get_test_func(x, func=None):
            try:
                x += 1
                try:
                    x /= 2
                    three_nesting(x, func)
                    x -= 3
                except ValueError:
                    x *= 4
            except:
                x += 5
            return x  # / 6

        self.assert_results(
            get_test_func, paddle.to_tensor(0.3), raise_value_error_obj
        )
        self.assert_results(
            get_test_func, paddle.to_tensor(0.4), raise_value_error_cls
        )
        self.assert_results(
            get_test_func, paddle.to_tensor(0.5), raise_zero_div_error
        )
        self.assert_results(
            get_test_func, paddle.to_tensor(0.6), raise_assert_error
        )
        self.assert_results(
            get_test_func, paddle.to_tensor(0.7), raise_not_implemented_error
        )


class TestAssertException(TestCaseBase):
    @staticmethod
    def try_assert(x, condition):
        # test py value or paddle tensor value as condition
        try:
            x += 1
            try:
                x /= 2
                raise TimeoutError("TESTING")
            except:
                x -= 3
                assert condition
        except:
            x *= 4
        return x / 5

    @strict_mode_guard(NOT_ALLOW_FALLBACK)
    def test_assert_with_py_var_as_condition(self):
        # Test the case where `condition` is Python variable
        self.assert_results(self.try_assert, paddle.to_tensor(1), False)
        self.assert_results(self.try_assert, paddle.to_tensor(2), True)
        self.assert_results(self.try_assert, paddle.to_tensor(3), [])
        self.assert_results(self.try_assert, paddle.to_tensor(4), [1])
        self.assert_results(self.try_assert, paddle.to_tensor(5), "")
        self.assert_results(self.try_assert, paddle.to_tensor(6), "QAQ")
        # TODO(DrRyanHuang): The following two cases are not supported yet.
        # self.assert_results(self.try_assert, paddle.to_tensor(7), ValueError)
        # self.assert_results(self.try_assert, paddle.to_tensor(8), ValueError())

    # Currently, since the assert statement is essentially an if statement and can cause breakgraph,
    # using a Tensor as a condition is not supported. Therefore, fallback is allowed.
    @strict_mode_guard(False)
    def test_assert_with_tensor_as_condition(self):
        # Test the case where `condition` is Paddle Tensor
        self.assert_results(
            self.try_assert, paddle.to_tensor(8), paddle.to_tensor(1)
        )
        self.assert_results(
            self.try_assert, paddle.to_tensor(9), paddle.to_tensor(0)
        )
        self.assert_results(
            self.try_assert, paddle.to_tensor(10), paddle.to_tensor(-1)
        )

    @strict_mode_guard(False)
    def test_assert_true(self):
        @check_no_breakgraph
        def try_assert_except(x):
            x += 1
            try:
                x += 2
                assert x > -10000
                x += 3
            except:
                x += 4

        self.assert_results(try_assert_except, paddle.to_tensor(10))

    @strict_mode_guard(False)
    def test_assert_false(self):
        @check_no_breakgraph
        def try_assert_except(x):
            try:
                x += 5
                assert x < -10000
            except AssertionError:
                x += 6

            return x

        self.assert_results(try_assert_except, paddle.to_tensor(10))


if __name__ == "__main__":
    unittest.main()
