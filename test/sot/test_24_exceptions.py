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
import unittest

from test_case_base import (
    TestCaseBase,
)

import paddle
from paddle.jit.sot import symbolic_translate
from paddle.jit.sot.opcode_translator.executor.opcode_executor import (
    ALREADY_SUPPORTED_EXCEPTION,
)
from paddle.jit.sot.psdb import check_no_breakgraph
from paddle.jit.sot.utils import strict_mode_guard

NOT_ALLOW_FALLBACK = ALREADY_SUPPORTED_EXCEPTION


class TestRaiseVarargs(TestCaseBase):
    # test `RAISE_VARARGS`

    @staticmethod
    @check_no_breakgraph
    def argc_equal_to_0_wo_exception(x):
        try:
            x += 1
            # In CPython, `RuntimeError` will be triggered at this point.
            # SOT have setup the exception stack and raise the exception manually.
            # This function is designed to test scenarios involving only the `raise` statement.
            raise  # Bare `raise` statement is not inside an exception handler # noqa: PLE0704
            x /= 2
        except RuntimeError:
            x -= 3
        x *= 4
        return x

    @staticmethod
    @check_no_breakgraph
    def argc_equal_to_0_zero_div_err(x):
        x += 1
        try:
            try:
                x += 2
                result = 10 / 0
            except ZeroDivisionError:
                x += 3
                raise  # RAISE_VARARGS(0)
        except:
            x += 4
        return x + 5

    @staticmethod
    @check_no_breakgraph
    def argc_equal_to_0_simulating_zero_div_err(x):
        x += 1
        try:
            try:
                x += 2
                raise ZeroDivisionError("")
            except ZeroDivisionError:
                x += 3
                raise  # RAISE_VARARGS(0)
        except:
            x += 4
        return x + 5

    @staticmethod
    @check_no_breakgraph
    def argc_equal_to_1(x):
        x += 1
        try:
            try:
                x += 2
            except:
                x += 3
            else:
                x += 4
                raise NotImplementedError  # RAISE_VARARGS(1)
                x += 5
        except:
            x += 6
        return x + 7

    @staticmethod
    @check_no_breakgraph
    def argc_equal_to_2(x):
        x -= 10
        try:
            try:
                x -= 300
            finally:
                x -= 400
                raise ValueError from None  # RAISE_VARARGS(2)
        except ValueError:
            x -= 500

        return x - 600

    @staticmethod
    @check_no_breakgraph
    def argc_equal_to_1_2(x):
        try:
            x += 1
            try:
                x /= 2
                raise NameError  # RAISE_VARARGS(1)
                x *= 3
            except NameError as e:
                x -= 4
                raise TimeoutError("TESTING") from e  # RAISE_VARARGS(2)
        except:
            x /= 5
        return x + 6

    @staticmethod
    @check_no_breakgraph
    def argc_equal_to_1_0(x):
        try:
            try:
                x -= 1
                raise ValueError("TESTING")  # RAISE_VARARGS(1)
                x += 2
            except NotImplementedError:
                x /= 3
                raise  # RAISE_VARARGS(0)
        except (KeyError, IndexError):
            x *= 4
        except ValueError:
            x += 5
        return x

    @strict_mode_guard(NOT_ALLOW_FALLBACK)
    def test_RAISE_VARARGS_argc(self):
        self.assert_results(
            self.argc_equal_to_0_wo_exception, paddle.to_tensor(0.01)
        )
        self.assert_results(
            self.argc_equal_to_0_zero_div_err, paddle.to_tensor(0.02)
        )
        self.assert_results(
            self.argc_equal_to_0_simulating_zero_div_err, paddle.to_tensor(0.03)
        )
        self.assert_results(self.argc_equal_to_1, paddle.to_tensor(0.04))
        self.assert_results(self.argc_equal_to_2, paddle.to_tensor(0.05))
        self.assert_results(self.argc_equal_to_1_2, paddle.to_tensor(0.06))
        self.assert_results(self.argc_equal_to_1_0, paddle.to_tensor(0.07))


class TestException(TestCaseBase):
    @staticmethod
    def create_builtin_exception(x):
        def identity(e):
            return e

        x += 1
        value_error = ValueError()
        identity(value_error)
        x += 2
        type_error = TypeError("")
        identity(type_error)
        x += 3
        key_error = KeyError("")
        identity(key_error)
        x += 4
        exception = Exception("")
        identity(exception)
        x += 5
        unicode_translate_error = UnicodeTranslateError("", -1, -1, "")
        identity(unicode_translate_error)
        x += 6
        return x

    @staticmethod
    def create_user_defined_exception(x):
        # TODO: Need to support user-defined exception
        return x

    @check_no_breakgraph
    @strict_mode_guard(NOT_ALLOW_FALLBACK)
    def test_dispatch(self):
        self.assert_results(
            self.create_builtin_exception, paddle.to_tensor(111.0)
        )
        self.assert_results(
            self.create_user_defined_exception, paddle.to_tensor(222.0)
        )


class TestTryExcept(TestCaseBase):
    # try ... except ...
    # ---------------- test raising exception directly ----------------
    @staticmethod
    def raise_value_error_obj():
        raise ValueError("Test whether raising `ValueError`")

    @staticmethod
    def raise_value_error_cls():
        raise ValueError

    @staticmethod
    def raise_asserterror(x):
        assert x, "Test AssertionError"

    # Since the exceptions are not handled, fallback is permitted.
    @strict_mode_guard(False)
    def test_exception_raising(self):
        with self.assertRaisesRegex(
            ValueError, "Test whether raising `ValueError`"
        ):
            symbolic_translate(self.raise_value_error_obj)()

        with self.assertRaisesRegex(ValueError, ""):
            symbolic_translate(self.raise_value_error_cls)()

        with self.assertRaisesRegex(AssertionError, "Test AssertionError"):
            symbolic_translate(self.raise_asserterror)(False)

        with self.assertRaisesRegex(AssertionError, "Test AssertionError"):
            symbolic_translate(self.raise_asserterror)(paddle.to_tensor(0))

    # ---------------- without error ----------------
    @staticmethod
    @check_no_breakgraph
    def try_except_wo_error(x):
        try:
            x = x + 1
        except:
            x = x * 2
        return x

    @staticmethod
    @check_no_breakgraph
    def try_except_exception_wo_error(x):
        try:
            x = x + 1
        except Exception:
            x = x * 2
        return x

    @staticmethod
    @check_no_breakgraph
    def try_except_exception_as_e_wo_error(x):
        try:
            x = x + 1
        except Exception as e:
            x = x * 2
        return x

    # ---------------- with error ----------------
    @staticmethod
    @check_no_breakgraph
    def try_except_with_error_obj(x):
        y = x + 3
        try:
            x = x + 1
            raise ValueError(f"{__class__.__name__}")
            x = x * 3
        except:
            y = x * 2
        return y

    @staticmethod
    @check_no_breakgraph
    def try_except_with_error_cls(x):
        y = x + 3
        try:
            x = x + 1
            raise ValueError
            x = x * 3
        except:
            y = x * 2
        return y

    @staticmethod
    @check_no_breakgraph
    def try_except_exception_with_error(x):
        # test `JUMP_IF_NOT_EXC_MATCH`
        x = x + 3
        try:
            x = x + 1
            raise ValueError("TESTING!")
            x = x * 3
        except Exception:
            x = x * 2
        return x

    @staticmethod
    @check_no_breakgraph
    def try_except_exception_as_e_with_error(x):
        # test `JUMP_IF_NOT_EXC_MATCH`
        y = x + 3
        try:
            x = x + 1
            raise ValueError("TESTING!")
            x = x * 3
        except ValueError as e:
            y = x * 2
        return y

    @staticmethod
    @check_no_breakgraph
    def try_except_exception_as_e_with_error_tuple(x):
        # test `JUMP_IF_NOT_EXC_MATCH`
        y = x + 3
        try:
            x = x + 1
            raise ValueError("TESTING!")
            x = x * 3
        except (ValueError, KeyError, NotImplementedError) as e:
            y = x * 2
        return y

    @staticmethod
    @check_no_breakgraph
    def try_except_exception_as_e_with_unmatched_error(x):
        # test `JUMP_IF_NOT_EXC_MATCH`
        y = x + 3
        try:
            x = x + 1
            raise ValueError("TESTING!")
            x = x * 3
        except KeyError as e:
            y = x * 2
        return y + 3

    @staticmethod
    @check_no_breakgraph
    def try_except_exception_as_e_with_matched_error_reraise(x):
        # test `JUMP_IF_NOT_EXC_MATCH`
        y = x + 3
        try:
            x = x + 1
            raise IndexError("TESTING!")
            x = x * 3
        except IndexError as e:
            y = x * 2
            raise LookupError("TESTING!")
        return y + 3

    @strict_mode_guard(NOT_ALLOW_FALLBACK)
    def test_try_except(self):
        self.assert_results(self.try_except_wo_error, paddle.to_tensor(2))
        self.assert_results(
            self.try_except_exception_wo_error, paddle.to_tensor(3)
        )
        self.assert_results(
            self.try_except_exception_as_e_wo_error, paddle.to_tensor(4)
        )
        self.assert_results(self.try_except_with_error_obj, paddle.to_tensor(5))
        self.assert_results(self.try_except_with_error_cls, paddle.to_tensor(6))
        self.assert_results(
            self.try_except_exception_with_error, paddle.to_tensor(7)
        )
        self.assert_results(
            self.try_except_exception_as_e_with_error, paddle.to_tensor(8)
        )
        self.assert_results(
            self.try_except_exception_as_e_with_error_tuple, paddle.to_tensor(9)
        )

    @strict_mode_guard(False)
    def test_error(self):
        # RERAISE
        self.assert_exceptions(
            ValueError,
            "TESTING!",
            self.try_except_exception_as_e_with_unmatched_error,
            paddle.to_tensor(0.001),
        )

        self.assert_exceptions(
            LookupError,
            "TESTING!",
            self.try_except_exception_as_e_with_matched_error_reraise,
            paddle.to_tensor(0.001),
        )


class TestTryFinally(TestCaseBase):
    # try ... finally ...
    # ---------------- without error ----------------
    @staticmethod
    @check_no_breakgraph
    def try_finally_wo_error(x):
        try:
            x = 1 + x
        finally:
            x *= 2
        return x

    # ---------------- with error ----------------
    @staticmethod
    @check_no_breakgraph
    def try_finally_with_error_but_return_in_finally(x):
        # RERAISE
        try:
            x = 3 + x
            raise NotImplementedError("TESTING!")
            x = 300 + x
        finally:
            x *= 2
            # `return` inside `finally` blocks cause exceptions to be silenced
            return x  # noqa: B012

    @staticmethod
    def try_finally_with_error(x):
        # RERAISE
        try:
            x = 3 + x
            raise NotImplementedError("TESTING!")
            x = 300 + x
        finally:
            x *= 2
        return x

    @staticmethod
    def try_finally_with_error_in_finally(x):
        # RERAISE
        try:
            x = 3 + x
            return x
        finally:
            x *= 2
            raise TimeoutError("TESTING!")
            x = 300 + x
        return x

    @strict_mode_guard(NOT_ALLOW_FALLBACK)
    def test_try_finally(self):
        self.assert_results(self.try_finally_wo_error, paddle.to_tensor(14))
        self.assert_results(
            self.try_finally_with_error_but_return_in_finally,
            paddle.to_tensor(15),
        )

    @strict_mode_guard(False)
    def test_error(self):
        # RERAISE
        self.assert_exceptions(
            NotImplementedError,
            "TESTING!",
            self.try_finally_with_error,
            paddle.to_tensor(16),
        )
        self.assert_exceptions(
            TimeoutError,
            "TESTING!",
            self.try_finally_with_error_in_finally,
            paddle.to_tensor(17),
        )


class TestTryExceptElse(TestCaseBase):
    # try ... except ... else
    # `else` is useful for code that must be executed if the try clause does not raise an exception.

    # ---------------- without error ----------------
    @staticmethod
    def try_except_else(x):
        try:
            x += 1
        except:
            x += 2
        else:
            x += 3
        return x

    # ---------------- with error ----------------
    @staticmethod
    def try_except_else_except_with_matched_error(x):
        try:
            x += 4
            raise ValueError
        except ValueError:
            x += 5
        else:
            x += 6
        return x

    @staticmethod
    @strict_mode_guard(False)
    def try_except_else_except_with_mismatched_error(x):
        try:
            x += 4
            raise TimeoutError
        except KeyError:
            x += 5
        else:
            x += 6
        return x

    @staticmethod
    def try_except_else_error_in_except(x):
        try:
            x += 4
        except KeyError:
            x += 5
            raise ValueError
        else:
            x += 6
        return x

    @staticmethod
    @strict_mode_guard(False)
    def try_except_else_error_in_else(x):
        try:
            x += 4
        except KeyError:
            x += 5
        else:
            x += 6
            raise ValueError("Testing!")
        return x

    @strict_mode_guard(NOT_ALLOW_FALLBACK)
    def test_try_except_else(self):
        # self.assert_results(self.try_except_else, paddle.to_tensor(14))
        self.assert_results(
            self.try_except_else_except_with_matched_error, paddle.to_tensor(15)
        )
        # self.assert_results(
        #     self.try_except_else_error_in_except, paddle.to_tensor(16)
        # )

    @strict_mode_guard(NOT_ALLOW_FALLBACK)
    def test_error(self):
        # RERAISE
        self.assert_exceptions(
            TimeoutError,
            "",
            self.try_except_else_except_with_mismatched_error,
            paddle.to_tensor(0.001),
        )
        self.assert_exceptions(
            ValueError,
            "Testing!",
            self.try_except_else_error_in_else,
            paddle.to_tensor(0.002),
        )


class TestTryExceptFinally(TestCaseBase):
    # try ... except ... finally
    # ---------------- without error ----------------
    @staticmethod
    def try_except_finally(x):
        try:
            x -= 1
        except:
            x -= 2
        finally:
            x -= 3
        return x

    # ---------------- without error ----------------
    @staticmethod
    def try_except_finally_with_matched_exception(x):
        try:
            x -= 1
            raise ValueError
        except:
            x -= 2
        finally:
            x -= 3
        return x

    @staticmethod
    @strict_mode_guard(False)
    def try_except_finally_with_mismatched_exception(x):
        try:
            x -= 1
            raise ValueError("TESTING")
        except AttributeError:
            x -= 2
        finally:
            x -= 3
        return x

    @staticmethod
    def try_except_finally_in_except(x):
        try:
            x -= 1
        except AttributeError:
            x -= 2
            raise ValueError
        finally:
            x -= 3
        return x

    @staticmethod
    @strict_mode_guard(False)
    def try_except_finally_in_finally(x):
        try:
            x -= 1
        except AttributeError:
            x -= 2
        finally:
            x -= 3
            raise ValueError
        return x

    @strict_mode_guard(NOT_ALLOW_FALLBACK)
    def test_try_except_finally(self):
        self.assert_results(self.try_except_finally, paddle.to_tensor([0.11]))
        self.assert_results(
            self.try_except_finally_with_matched_exception,
            paddle.to_tensor([0.22]),
        )
        self.assert_results(
            self.try_except_finally_in_except,
            paddle.to_tensor([0.33]),
        )

    @strict_mode_guard(NOT_ALLOW_FALLBACK)
    def test_error(self):
        # RERAISE
        self.assert_exceptions(
            ValueError,
            "TESTING",
            self.try_except_finally_with_mismatched_exception,
            paddle.to_tensor(0.001),
        )

        self.assert_exceptions(
            ValueError,
            "",
            self.try_except_finally_in_finally,
            paddle.to_tensor(0.001),
        )


class TestTryExceptElseFinally(TestCaseBase):
    # try ... except ... else ... finally
    # ---------------- without error ----------------
    @staticmethod
    def try_except_else_finally(x):
        try:
            x -= 1
        except:
            x -= 2
        else:
            x -= 3
        finally:
            x -= 4
        return x

    # ---------------- with error ----------------
    @staticmethod
    def try_except_else_finally_with_matched_exception(x):
        try:
            x -= 1
            raise ValueError
        except ValueError:
            x -= 2
        else:
            x -= 3
        finally:
            x -= 4
        return x

    @staticmethod
    @strict_mode_guard(False)
    def try_except_else_finally_with_mismatched_exception(x):
        try:
            x -= 1
            raise SystemError("TESTING")
        except NotImplementedError:
            x -= 2
        else:
            x -= 3
        finally:
            x -= 4
        return x

    @staticmethod
    @strict_mode_guard(False)
    def try_except_else_finally_with_exception_in_else(x):
        try:
            x -= 1
        except NotImplementedError:
            x -= 2
        else:
            x -= 3
            raise ModuleNotFoundError("TESTING")
        finally:
            x -= 4
        return x

    @staticmethod
    @strict_mode_guard(False)
    def try_except_else_finally_with_exception_in_finally(x):
        try:
            x -= 1
        except NotImplementedError:
            x -= 2
        else:
            x -= 3
        finally:
            x -= 4
            raise SyntaxError("TESTING")
        return x

    @strict_mode_guard(NOT_ALLOW_FALLBACK)
    def test_try_except_finally(self):
        self.assert_results(
            self.try_except_else_finally, paddle.to_tensor([0.11])
        )
        self.assert_results(
            self.try_except_else_finally_with_matched_exception,
            paddle.to_tensor([0.22]),
        )

    @strict_mode_guard(NOT_ALLOW_FALLBACK)
    def test_error(self):
        # RERAISE
        self.assert_exceptions(
            SystemError,
            "TESTING",
            self.try_except_else_finally_with_mismatched_exception,
            paddle.to_tensor(0.001),
        )
        self.assert_exceptions(
            ModuleNotFoundError,
            "TESTING",
            self.try_except_else_finally_with_exception_in_else,
            paddle.to_tensor(0.001),
        )
        self.assert_exceptions(
            SyntaxError,
            "TESTING",
            self.try_except_else_finally_with_exception_in_finally,
            paddle.to_tensor(0.001),
        )


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


class TestGuard(TestCaseBase):
    @strict_mode_guard(False)
    @check_no_breakgraph
    def test_guard_run(self):
        def fn():
            try:
                paddle.jit.sot.psdb.breakgraph()
                raise ValueError
            except ValueError:
                return True
            return False

        self.assert_results(fn)


class TestBuiltinFunctionRaiseExceptionGuard(TestCaseBase):
    def test_guard_run(self):
        def foo_floordiv(x):
            1 / x

        def foo_mod(x):
            2 % x

        self.assert_results(foo_floordiv, 1)
        self.assert_exceptions(
            ZeroDivisionError,
            "division by zero",
            foo_floordiv,
            0,
        )
        self.assert_results(foo_mod, 10)
        self.assert_exceptions(
            ZeroDivisionError,
            "integer (.)*modulo by zero",
            foo_mod,
            0,
        )


if __name__ == "__main__":
    unittest.main()
