# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import os
import unittest

from paddle.utils.environments import (
    BooleanEnvironmentVariable,
    EnvironmentVariableGuard,
    IntegerEnvironmentVariable,
    StringEnvironmentVariable,
)


class TestBooleanEnvironmentVariable(unittest.TestCase):
    def test_bool_env_get(self):
        env_name = "___TEST_ENV_BOOL_GET"
        env_bool = BooleanEnvironmentVariable(env_name, False)

        self.assertIs(env_bool.get(), False)

        os.environ[env_name] = "False"
        self.assertIs(env_bool.get(), False)

        os.environ[env_name] = "OFF"
        self.assertIs(env_bool.get(), False)

        os.environ[env_name] = "0"
        self.assertIs(env_bool.get(), False)

        os.environ[env_name] = "True"
        self.assertIs(env_bool.get(), True)

        os.environ[env_name] = "ON"
        self.assertIs(env_bool.get(), True)

        os.environ[env_name] = "1"
        self.assertIs(env_bool.get(), True)

    def test_bool_env_set(self):
        env_name = "___TEST_ENV_BOOL_SET"
        env_bool = BooleanEnvironmentVariable(env_name, False)

        env_bool.set(True)
        self.assertIs(env_bool.get(), True)

        env_bool.set(False)
        self.assertIs(env_bool.get(), False)

        with self.assertRaises(AssertionError):
            env_bool.set("True")

        with self.assertRaises(AssertionError):
            env_bool.set("False")

        with self.assertRaises(AssertionError):
            env_bool.set(0)

        with self.assertRaises(AssertionError):
            env_bool.set(1)

    def test_bool_env_guard(self):
        env_name = "___TEST_ENV_BOOL_GUARD"
        env_bool = BooleanEnvironmentVariable(env_name, False)

        with EnvironmentVariableGuard(env_bool, True):
            self.assertIs(env_bool.get(), True)

        with EnvironmentVariableGuard(env_bool, False):
            self.assertIs(env_bool.get(), False)


class TestStringEnvironmentVariable(unittest.TestCase):
    def test_str_env_get(self):
        env_name = "___TEST_ENV_STR_GET"
        env_str = StringEnvironmentVariable(env_name, "DEFAULT")

        self.assertEqual(env_str.get(), "DEFAULT")

        os.environ[env_name] = "CASE1"
        self.assertEqual(env_str.get(), "CASE1")

        os.environ[env_name] = "CASE2"
        self.assertEqual(env_str.get(), "CASE2")

    def test_str_env_set(self):
        env_name = "___TEST_ENV_STR_SET"
        env_str = StringEnvironmentVariable(env_name, "DEFAULT")

        self.assertEqual(env_str.get(), "DEFAULT")

        env_str.set("CASE1")
        self.assertEqual(env_str.get(), "CASE1")

        env_str.set("CASE2")
        self.assertEqual(env_str.get(), "CASE2")

        with self.assertRaises(AssertionError):
            env_str.set(True)

        with self.assertRaises(AssertionError):
            env_str.set(False)

        with self.assertRaises(AssertionError):
            env_str.set(0)

        with self.assertRaises(AssertionError):
            env_str.set(1)

    def test_str_env_guard(self):
        env_name = "___TEST_ENV_STR_GUARD"
        env_str = StringEnvironmentVariable(env_name, "DEFAULT")

        with EnvironmentVariableGuard(env_str, "CASE1"):
            self.assertEqual(env_str.get(), "CASE1")

        with EnvironmentVariableGuard(env_str, "CASE2"):
            self.assertEqual(env_str.get(), "CASE2")


class TestIntegerEnvironmentVariable(unittest.TestCase):
    def test_int_env_get(self):
        env_name = "___TEST_ENV_INT_GET"
        env_int = IntegerEnvironmentVariable(env_name, 42)

        self.assertEqual(env_int.get(), 42)

        os.environ[env_name] = "10"
        self.assertEqual(env_int.get(), 10)

        os.environ[env_name] = "99999"
        self.assertEqual(env_int.get(), 99999)

    def test_int_env_set(self):
        env_name = "___TEST_ENV_INT_SET"
        env_int = IntegerEnvironmentVariable(env_name, 42)

        self.assertEqual(env_int.get(), 42)

        env_int.set(99)
        self.assertEqual(env_int.get(), 99)

        env_int.set(1000)
        self.assertEqual(env_int.get(), 1000)

        with self.assertRaises(AssertionError):
            env_int.set(True)

        with self.assertRaises(AssertionError):
            env_int.set(False)

        with self.assertRaises(AssertionError):
            env_int.set("10")

        with self.assertRaises(AssertionError):
            env_int.set("42")

    def test_int_env_guard(self):
        env_name = "___TEST_ENV_INT_GUARD"
        env_int = IntegerEnvironmentVariable(env_name, 42)

        with EnvironmentVariableGuard(env_int, 99):
            self.assertEqual(env_int.get(), 99)

        with EnvironmentVariableGuard(env_int, 1000):
            self.assertEqual(env_int.get(), 1000)


if __name__ == "__main__":
    unittest.main()
