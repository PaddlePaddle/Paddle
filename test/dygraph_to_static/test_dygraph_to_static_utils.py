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
from itertools import product

from dygraph_to_static_utils import (
    DEFAULT_BACKEND_MODE,
    DEFAULT_IR_MODE,
    DEFAULT_TO_STATIC_MODE,
    VALID_MODES,
    BackendMode,
    Dy2StTestBase,
    Dy2StTestMeta,
    IrMode,
    ModeTuple,
    ToStaticMode,
    disable_test_case,
    set_backend_mode,
    set_ir_mode,
    set_to_static_mode,
)

ALL_MODES = list(product(ToStaticMode, IrMode, BackendMode))
DEFAULT_MODES = [
    (to_static_mode, ir_mode, backend_mode)
    for (to_static_mode, ir_mode, backend_mode) in ALL_MODES
    if (
        (to_static_mode, ir_mode, backend_mode) in VALID_MODES
        and to_static_mode & DEFAULT_TO_STATIC_MODE
        and ir_mode & DEFAULT_IR_MODE
        and backend_mode & DEFAULT_BACKEND_MODE
    )
]


class CheckTestCaseExistsMixin:
    def assert_hasattr(self, obj: object, attr: str):
        self.assertTrue(hasattr(obj, attr), msg=f"{attr} not in {obj.__dict__.keys()}")  # type: ignore

    def assert_not_hasattr(self, obj: object, attr: str):
        self.assertFalse(hasattr(obj, attr), msg=f"{attr} in {obj.__dict__.keys()}")  # type: ignore

    def check_test_case_exists(
        self, test_case: Dy2StTestBase, case_name: str, mode_tuple: ModeTuple
    ):
        new_case_name = Dy2StTestMeta.test_case_name(case_name, mode_tuple)
        self.assert_hasattr(test_case, new_case_name)

    def check_test_case_not_exists(
        self, test_case: Dy2StTestBase, case_name: str, mode_tuple: ModeTuple
    ):
        new_case_name = Dy2StTestMeta.test_case_name(case_name, mode_tuple)
        self.assert_not_hasattr(test_case, new_case_name)


class TestCaseBasic(Dy2StTestBase):
    def test_basic(self): ...


class TestCaseDisableTestCase(Dy2StTestBase):
    @disable_test_case((ToStaticMode.SOT, IrMode.PIR, BackendMode.CINN))
    def test_disable_one(self): ...

    @disable_test_case((ToStaticMode.SOT, IrMode.PIR, BackendMode.CINN))
    @disable_test_case((ToStaticMode.SOT, IrMode.PIR, BackendMode.PHI))
    @disable_test_case((ToStaticMode.AST, IrMode.PIR, BackendMode.PHI))
    def test_disable_multiple(self): ...

    @disable_test_case(
        (ToStaticMode.SOT, IrMode.PIR, BackendMode.CINN | BackendMode.PHI)
    )
    def test_disable_multiple_with_or(self): ...


class TestCaseSetMode(Dy2StTestBase):
    @set_to_static_mode(ToStaticMode.SOT)
    def test_set_to_static_mode(self): ...

    @set_ir_mode(IrMode.PIR)
    def test_set_ir_mode(self): ...

    @set_backend_mode(BackendMode.CINN)
    def test_set_backend_mode(self): ...

    @set_to_static_mode(ToStaticMode.SOT)
    @set_ir_mode(IrMode.PIR)
    @set_backend_mode(BackendMode.CINN)
    def test_set_all(self): ...


class TestCheckTestCases(unittest.TestCase, CheckTestCaseExistsMixin):
    def test_check_test_case_basic(self):
        test_case = TestCaseBasic()
        case_name = "test_basic"
        self.assert_not_hasattr(test_case, case_name)
        for mode_tuple in DEFAULT_MODES:
            self.check_test_case_exists(test_case, case_name, mode_tuple)

    def test_check_test_case_disable_test_case(self):
        test_case = TestCaseDisableTestCase()
        case_name = "test_disable_one"
        self.assert_not_hasattr(test_case, case_name)
        for mode_tuple in DEFAULT_MODES:
            if mode_tuple == (ToStaticMode.SOT, IrMode.PIR, BackendMode.CINN):
                self.check_test_case_not_exists(
                    test_case, case_name, mode_tuple
                )
            else:
                self.check_test_case_exists(test_case, case_name, mode_tuple)

        case_name = "test_disable_multiple"
        self.assert_not_hasattr(test_case, case_name)
        for mode_tuple in DEFAULT_MODES:
            if mode_tuple in [
                (ToStaticMode.SOT, IrMode.PIR, BackendMode.CINN),
                (ToStaticMode.SOT, IrMode.PIR, BackendMode.PHI),
                (ToStaticMode.AST, IrMode.PIR, BackendMode.PHI),
            ]:
                self.check_test_case_not_exists(
                    test_case, case_name, mode_tuple
                )
            else:
                self.check_test_case_exists(test_case, case_name, mode_tuple)

        case_name = "test_disable_multiple_with_or"
        self.assert_not_hasattr(test_case, case_name)
        for mode_tuple in DEFAULT_MODES:
            if mode_tuple in [
                (ToStaticMode.SOT, IrMode.PIR, BackendMode.CINN),
                (ToStaticMode.SOT, IrMode.PIR, BackendMode.PHI),
            ]:
                self.check_test_case_not_exists(
                    test_case, case_name, mode_tuple
                )
            else:
                self.check_test_case_exists(test_case, case_name, mode_tuple)

    def test_check_test_case_set_mode(self):
        test_case = TestCaseSetMode()
        case_name = "test_set_to_static_mode"
        self.assert_not_hasattr(test_case, case_name)
        for mode_tuple in DEFAULT_MODES:
            to_static_mode, _, _ = mode_tuple
            if to_static_mode == ToStaticMode.SOT:
                self.check_test_case_exists(test_case, case_name, mode_tuple)
            else:
                self.check_test_case_not_exists(
                    test_case, case_name, mode_tuple
                )

        case_name = "test_set_ir_mode"
        self.assert_not_hasattr(test_case, case_name)
        for mode_tuple in DEFAULT_MODES:
            _, ir_mode, _ = mode_tuple
            if ir_mode == IrMode.PIR:
                self.check_test_case_exists(test_case, case_name, mode_tuple)
            else:
                self.check_test_case_not_exists(
                    test_case, case_name, mode_tuple
                )

        case_name = "test_set_backend_mode"
        self.assert_not_hasattr(test_case, case_name)
        for mode_tuple in DEFAULT_MODES:
            _, _, backend_mode = mode_tuple
            if backend_mode == BackendMode.CINN:
                self.check_test_case_exists(test_case, case_name, mode_tuple)
            else:
                self.check_test_case_not_exists(
                    test_case, case_name, mode_tuple
                )

        case_name = "test_set_all"
        self.assert_not_hasattr(test_case, case_name)
        for mode_tuple in DEFAULT_MODES:
            if mode_tuple == (ToStaticMode.SOT, IrMode.PIR, BackendMode.CINN):
                self.check_test_case_exists(test_case, case_name, mode_tuple)
            else:
                self.check_test_case_not_exists(
                    test_case, case_name, mode_tuple
                )


if __name__ == "__main__":
    unittest.main()
