# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.distributed.passes.pass_base import register_pass, PassBase, new_pass
from paddle.distributed.passes.pass_base import _make_rule_from_white_lists_dict as make_white_lists_rule


class TestConcretePass(PassBase):
    def __init__(self):
        super(TestConcretePass, self).__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        pass


@register_pass("A")
class A(TestConcretePass):
    def __init__(self):
        super(A, self).__init__()


@register_pass("B")
class B(TestConcretePass):
    def __init__(self):
        super(B, self).__init__()


@register_pass("C")
class C(TestConcretePass):
    def __init__(self):
        super(C, self).__init__()


@register_pass("D")
class D(TestConcretePass):
    def __init__(self):
        super(D, self).__init__()


@register_pass("E")
class E(TestConcretePass):
    def __init__(self):
        super(E, self).__init__()


class TestMakeWhiteListsRule(unittest.TestCase):
    def test_main(self):
        before_white_lists = {"A": ["B", "C"]}
        after_white_lists = {"D": ["C"]}
        rule = make_white_lists_rule(before_white_lists, after_white_lists)

        pass_a = new_pass("A")
        pass_b = new_pass("B")
        pass_c = new_pass("C")
        pass_d = new_pass("D")
        pass_e = new_pass("E")

        self.assertTrue(rule(pass_a, pass_e))
        self.assertTrue(rule(pass_e, pass_a))

        self.assertTrue(rule(pass_a, pass_b))
        self.assertFalse(rule(pass_b, pass_a))
        self.assertTrue(rule(pass_a, pass_c))
        self.assertFalse(rule(pass_c, pass_a))

        self.assertFalse(rule(pass_a, pass_d))
        self.assertFalse(rule(pass_d, pass_a))

        self.assertTrue(rule(pass_c, pass_d))
        self.assertFalse(rule(pass_d, pass_c))


if __name__ == "__main__":
    unittest.main()
